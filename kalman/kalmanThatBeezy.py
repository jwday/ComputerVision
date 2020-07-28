# Updated 5/5/2020
# THIS IS INCOMPLETE AS OF 5/5/2020
# This function computes the estimated state of a process using Kalman filtering.
# The state measurements must be supplied via .csv as an input parameter when executing the script.
# i.e.
# > $ ipython 3D_Kalman_specify_loc.py ~/DESKTOP/Photos/datafile.csv

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from mpl_toolkits.mplot3d import Axes3D
from filterpy.kalman import KalmanFilter
import os
import sys

datafile = './datafile.csv'			# Default data file, used for development

def estimate_pose(datafile, delim_whitespace=False):
	datafile_name = datafile.split('/')[-1].split('.')[0]
	valvegroup = datafile_name.split('_')[0]
	movement_type = datafile_name.split('_')[1]

	df = pd.read_csv(datafile, header=0)								# Columns should already be specified as ['Time (s)', 'r1', 'r2', 'r3', 't1', 't2', 't3'] when the datafile is written.
	df.dropna(axis=0, how='any', inplace=True)							# Drop all the rows that contain no data (i.e. Aruco marker detection failed)
	df.reset_index(drop=True, inplace=True)								# Reset the indices so the Kalman filter will iterate properly over the dataframe

	# ================================================================
	# HANDLE TIME DATA
	# ================================================================
	df['dt'] = df['Time (s)'].diff()									# Add a column of time differentials (dt) between measurements
	df['dt'] = df['dt'].fillna(df['dt'].mean())							# Fill any NA's with the average dt


	# ================================================================
	# HANDLE TRANSLATION DATA
	# ================================================================
	df['t1'] = [x - df['t1'][0] for x in df['t1']]						# Offset by initial value (zero the data)
	df['t2'] = [x - df['t2'][0] for x in df['t2']]						# Offset by initial value (zero the data)
	df['t2'] = -df['t2']												# Because opencv returns the translation FROM the Aruco marker TO the camera, but we want to see it from the other side.
	
	df['tt'] = (df['t1'] ** 2 + df['t2'] ** 2)**0.5

	 									# Total position (sqrt(x^2 + y^2))
						
										#	Note that this can only occur after zeroing data


	# ================================================================
	# HANDLE ROTATION DATA
	# ================================================================
	df['r1'] = -np.rad2deg(df['r1'])									# Because opencv returns the translation FROM the Aruco marker TO the camera, but we want to see it from the other side. Also convert to degrees.
	df['r1'] = [x - df['r1'][0] for x in df['r1']]						# Offset by initial value

	add_amount = 0														# This block will handle incidents when the angle jumps from -180 to +180
	temp_list = list(df['r1'])											#
	for i, x in enumerate(df['r1'][1:]):								#
		diff = x - df['r1'][i]											#
		if diff >= 175:													#
			add_amount -= 180											#
			for j, y in enumerate(df['r1'][i+1:]):						#
				temp_list[j+i+1] = y + add_amount						#
		if diff <= -175:												#
			add_amount += 180											#
			for j, y in enumerate(df['r1'][i+1:]):						#
				temp_list[j+i+1] = y + add_amount						#
	df['r1'] = temp_list												#


	# ================================================================
	# BEGIN DOING THE KALMAN FILTER THING
	# ================================================================
	zs = df[['t1', 't2', 'r1']].values									# Create a Numpy array of the measurements for use in the Kalman filter


	# -----------------------------------------------------------------
	# STATE COVARIANCE MATRIX (P)
	# -----------------------------------------------------------------
	# P = np.array([[9.0E-6, 0, 0], [0, 9.0E-6, 0], [0, 0, 9.0E-6]])	# Initial state covariance (expected variance of each state variable). Should be of size NxN for N tracked states.
	# P = np.eye(6)*9.0E-6
	P = np.array([[9.0E-4,		 0,		 0,		 0,		 0,		 0,		 0,		 0,		 0],  # x
				  [		0,	9.0E-2,		 0,		 0,		 0,		 0,		 0,		 0,		 0],  # xdot
				  [		0,		 0,	9.0E-0,		 0,		 0,		 0,		 0,		 0,		 0],  # xdotdot
				  [		0,		 0,		 0,	9.0E-4,		 0,		 0,		 0,		 0,		 0],  # y
				  [		0,		 0,		 0,		 0,	9.0E-2,		 0,		 0,		 0,		 0],  # ydot
				  [		0,		 0,		 0,		 0,		 0,	9.0E-0,		 0,		 0,		 0],  # ydotdot
				  [		0,		 0,		 0,		 0,		 0,		 0,	9.0E-4,		 0,		 0],  # r1
				  [		0,		 0,		 0,		 0,		 0,		 0,		 0, 9.0E-2,		 0],  # r1dot
				  [		0,		 0,		 0,		 0,		 0,		 0,		 0,		 0,	9.0E-0]]) # r1dotdot


	# -----------------------------------------------------------------
	# PROCESS NOISE COVARIANCE MATRIX (Q)
	# -----------------------------------------------------------------
	# Q = np.array([[9.0E-6, 0, 0], [0, 9.0E-6, 0], [0, 0, 9.0E-6]])	# Process noise covariance. This is different than state covariance and remains UNCHANGED through the Kalman filter. Should be of size NxN for N tracked states.
	# Q = np.eye(6)*9.0E-6

	# FilterPy provides functions which computes Q by calculating the discrete-time white noise
	# Q_discrete_white_noise takes 4 parameters:
	# ..... "dim"; 	Specifies the dimension of the matrix.
	# .....	"dt"; 	Time step in seconds (if non-constant, take an average).
	# .....	"var"; 	White noise variance.
	# .....	"block_size": If your state variable contains more than one dimension, such as a 3d constant velocity model [x x’ y y’ z z’]^T, then Q must be a block diagonal matrix.
	# For a 2d constant acceleration model [x  x' x'' y  y' y''], dim=3, block_size=2.
	# For a 3d constant acceleration model with rotation, [x  x' x'' y  y' y'' r  r' r''], dim=3, block_size=3.

	from filterpy.common import Q_discrete_white_noise
	Q_t1 = Q_discrete_white_noise(dim=3, dt=df['dt'].mean(), var=2.35, block_size=1)	# White noise for X pos
	Q_t2 = Q_t1																			# White noise for Y pos
	Q_r1 = Q_discrete_white_noise(dim=3, dt=df['dt'].mean(), var=0.1, block_size=1)		# Different white noise for Theta
	Q = linalg.block_diag(Q_t1, Q_t2, Q_r1)

	# -----------------------------------------------------------------
	# MEASUREMENT COVARIANCE MATRIX (R)
	# -----------------------------------------------------------------
	R = np.array([[1.0E0, 		 0,			0],				# Measurement variance/covariance. Should be size MxM for M measured states. Each value is the variance/covariance of the state measurement.
				  [	 0,		1.0E0,		 	0],
				  [	 0,   		 0, 	  	1]])


	# -----------------------------------------------------------------
	# STATE-SPACE TO MEASUREMENT-SPACE CONVERSION MATRIX (H)
	# -----------------------------------------------------------------
	H = np.array([[1., 	0, 	0, 	0, 	0, 	0,	0,	0,	0],					# Measurement-space conversion. Should be size MxN for M measured states and N tracked states.
				  [ 0, 	0, 	0, 	1, 	0, 	0,	0,	0,	0],					# Measured states are associated with a 1 (or whatever conversion factor is necessary to translate the state into an associated measurement). Everything else is 0.
				  [ 0, 	0, 	0, 	0, 	0, 	0,	1,	0,	0]])				# In this case, we're measuring position but not velocity or acceleration, so [1, 0, 0] such that Hx yields only position predictions in the residual.


	# -----------------------------------------------------------------
	# STATE TRANSITION MODEL MATRIX (F)
	# -----------------------------------------------------------------
	def state_transition_matrix(k=None):								# State transition matrix of the system, to be updated every time the function is called because dt is not constant. Should be of size NxN for N tracked states.
		dt = df['dt'][k]
		F = np.array([[1, 	dt,	0.5*dt**2, 		0, 		0,		   0,		0,		0,			0],	 # x
					  [0, 	 1, 	   dt, 		0, 		0,		   0,		0,		0,			0],	 # xdot
					  [0, 	 0, 		1, 		0, 		0,		   0,		0,		0,			0],  # xdotdot
					  [0, 	 0, 		0, 		1, 	   dt, 0.5*dt**2,		0,		0,			0],  # y
					  [0, 	 0, 		0, 		0, 		1, 		  dt,		0,		0,			0],  # ydot
					  [0, 	 0, 		0, 		0, 		0, 		   1,		0,		0,			0],	 # ydotdot
					  [0, 	 0, 		0, 		0, 		0, 		   0,		1,		dt,	0.5*dt**2],	 # r1
					  [0, 	 0, 		0, 		0, 		0, 		   0,		0,		1,		   dt],	 # r1dot
					  [0, 	 0, 		0, 		0, 		0, 		   0,		0,		0,			1]]) # r1dotdot
		return F


	# -----------------------------------------------------------------
	# CONTROL INPUT MATRIX (B)
	# -----------------------------------------------------------------
	# B = np.array([0, 0, 0.02])			# Optional control input model
	# u = 0									# Optional control inputs


	# -----------------------------------------------------------------
	# INITIAL STATE MATRIX (x)
	# -----------------------------------------------------------------
	# Initial state. If I don't know what the initial state is, just set it to the first measurement.
	# x = np.array([df['t1'][0],  # x
	# 			  (-3*df['t1'][0] + 4*df['t1'][1] - df['t1'][2])/(df['dt'][0] + df['dt'][1]),  # xdot (first derivative, second-order accurate)
	# 			  (-2*df['t1'][0] - 3*df['t1'][1] + 6*df['t1'][2] - df['t1'][3])/(6*df['dt'][0]),  # xdotdot (second derivative, third-order accurate)
	# 			  df['t2'][0],  # y
	# 			  (-3*df['t2'][0] + 4*df['t2'][1] - df['t2'][2])/(df['dt'][0] + df['dt'][1]),  # ydot (first derivative, second-order accurate)
	# 			  (-2*df['t2'][0] - 3*df['t2'][1] + 6*df['t2'][2] - df['t2'][3])/(6*df['dt'][0]),  # ydotdot (second derivative, third-order accurate)
	# 			  df['r1'][0],  # r1
	# 			  		 0,  # r1dot
	# 			  		 0]) # r1dotdot
	# I offset all init values to zero and the motion begins at zero so make it zero
	x = np.zeros(9)


	# -----------------------------------------------------------------
	# KALMAN FILTER
	# -----------------------------------------------------------------
	# Set up the Kalman Filter object
	kf = KalmanFilter(dim_x=9, dim_z=3, dim_u=0) 	# Initialize a KalmanFilter object.
	kf.x = x 										# Specify initial state.
	kf.F = state_transition_matrix(k=0) 			# Specify initial state transition matrix.
	kf.H = H					   					# Measurement function, to define what states are being measured.
	kf.R *= R                     					# Measurement variance/covariance matrix.
	kf.P[:] = P               						# Specify initial state covariance matrix. [:] makes deep copy. Not sure what that means.
	kf.Q[:] = Q										# Specify initial process covariance matrix. [:] makes deep copy. Not sure what that means.

	# Run the Kalman filter and store the results
	xs, cov = [], []
	for i, z in enumerate(zs):						# Loop through the measurements.
		kf.F = state_transition_matrix(i)			# For each measurement, update F (the state transition matrix).
		kf.predict()								# Predict the next state based on the current state and the model.
		kf.update(z)								# Update the prediction using the measurements from that state (this is where the heavy lifting is done in the Kalman filter).
		xs.append(kf.x)								# Add the new estimation to a list.
		cov.append(kf.P.flatten())					# Add the new state covariance to a list.

	# Organize the results
	x_filt = pd.DataFrame(xs)						# Make a Pandas DataFrame from the Kalman-estimated states.
	x_filt.columns = ['X Position', 'X Velocity', 'X Acceleration', 'Y Position', 'Y Velocity', 'Y Acceleration', 'r1 Angle', 'r1 Velocity', 'r1 Acceleration']
	# x_filt['Time (s)'] = df['Time (s)']
	# x_filt['XY Position'] = x_filt.apply(lambda row: np.sqrt((row['X Position'] - x_filt['X Position'][0])**2 + (row['Y Position'] - x_filt['Y Position'][0])**2), axis=1)
	# x_filt['XY Velocity'] = x_filt.apply(lambda row: np.sqrt((row['X Velocity'] - x_filt['X Velocity'][0])**2 + (row['Y Velocity'] - x_filt['Y Velocity'][0])**2), axis=1)
	# x_filt['XY Acceleration'] = x_filt.apply(lambda row: np.sqrt((row['X Acceleration'] - x_filt['X Acceleration'][0])**2 + (row['Y Acceleration'] - x_filt['Y Acceleration'][0])**2), axis=1)
	x_filt['XY Position'] = (x_filt['X Position']**2 + x_filt['Y Position']**2)**0.5
	x_filt['XY Velocity'] = (x_filt['X Velocity']**2 + x_filt['Y Velocity']**2)**0.5
	x_filt['XY Acceleration'] = (x_filt['X Acceleration']**2 + x_filt['Y Acceleration']**2)**0.5

	total_time = np.round(df['Time (s)'].iloc[-1],1)
	framerate = np.round(df.count()[0]/df['Time (s)'].iloc[-1], 2)
	
	

	# -----------------------------------------------------------------
	# PLOT RESULTS
	# -----------------------------------------------------------------
	# # Plot just Rotation
	# fig1, ax = plt.subplots(3, sharex=True, dpi=150, figsize=[7, 5])
	# fig1.suptitle('{} Failure Mode During {}, Angular Displacement'.format(valvegroup, movement_type), y=0.97, fontsize=12)
	# # ax[0].set_title('cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX', fontsize=10)
	# df.plot(ax=ax[0],
	# 		x='Time (s)', y='r1',
	# 		marker='o',
	# 		markersize=2,
	# 		linestyle='None',
	# 		label='Measured Angle')
	# x_filt.plot(ax=ax[0], x='Time (s)', y='r1 Angle', color ='#ff7f0e', label='Estimated Angle')
	# x_filt.plot(ax=ax[1], x='Time (s)', y='r1 Velocity', color ='#ff7f0e', label='Estimated Angular Velocity')
	# x_filt.plot(ax=ax[2], x='Time (s)', y='r1 Acceleration', color ='#ff7f0e', label='Estimated Angular Acceleration')
	# ax[0].legend(loc='best')
	# ax[1].legend(loc='best')
	# ax[2].legend(loc='best')
	# ax[0].set_ylabel(r'$\Theta$ (deg)', color='#413839', fontsize=10)
	# ax[1].set_ylabel(r'$\omega$ (deg/s)', color='#413839', fontsize=10)
	# ax[2].set_ylabel(r'$\alpha$ (deg/$s^2$)', color='#413839', fontsize=10)


	# # # Plot just X
	# fig2, ax = plt.subplots(3, sharex=True, dpi=150, figsize=[7, 5])
	# plt.suptitle('{} Failure Mode During {}, X-axis'.format(valvegroup, movement_type), y=0.95, fontsize=12)
	# df.plot(ax=ax[0],
	# 		x='Time (s)', y='t1',
	# 		marker='o',
	# 		markersize=2,
	# 		linestyle='None',
	# 		label='Measured')
	# x_filt.plot(ax=ax[0], x='Time (s)', y='X Position', color ='#ff7f0e', label='Estimated Position')
	# x_filt.plot(ax=ax[1], x='Time (s)', y='X Velocity', color ='#ff7f0e', label='Estimated Velocity')
	# x_filt.plot(ax=ax[2], x='Time (s)', y='X Acceleration', color ='#ff7f0e', label='Estimated Acceleration')
	# ax[0].legend(loc='upper left')
	# ax[1].legend(loc='upper right')
	# ax[2].legend(loc='upper right')
	# ax[0].set_ylabel(r'$d_x$ (m)', color='#413839', fontsize=10)
	# ax[1].set_ylabel(r'$\dot{d_x}$ (m/s)', color='#413839', fontsize=10)
	# ax[2].set_ylabel(r'$\ddot{d_x}$ (m/$s^2$)', color='#413839', fontsize=10)


	# # # Plot just Y
	# fig3, ax = plt.subplots(3, sharex=True, dpi=150, figsize=[7, 5])
	# plt.suptitle('{} Failure Mode During {}, Y-axis'.format(valvegroup, movement_type), y=0.95, fontsize=12)
	# df.plot(ax=ax[0],
	# 		x='Time (s)', y='tt',
	# 		marker='o',
	# 		markersize=2,
	# 		linestyle='None',
	# 		label='r1 Measured')
	# x_filt.plot(ax=ax[0], x='Time (s)', y='XY Position', color ='#ff7f0e', label='Estimated Position')
	# x_filt.plot(ax=ax[1], x='Time (s)', y='XY Velocity', color ='#ff7f0e', label='Estimated Velocity')
	# x_filt.plot(ax=ax[2], x='Time (s)', y='Y Acceleration', color ='#ff7f0e', label='Estimated Acceleration')
	# ax[0].legend(loc='upper left')
	# ax[1].legend(loc='upper right')
	# ax[2].legend(loc='upper right')
	# ax[0].set_ylabel(r'$d_y$ (m)', color='#413839', fontsize=10)
	# ax[1].set_ylabel(r'$\dot{d_y}$ (m/s)', color='#413839', fontsize=10)
	# ax[2].set_ylabel(r'$\ddot{d_y}$ (m/$s^2$)', color='#413839', fontsize=10)


	# # Plot XY
	# fig4, ax = plt.subplots(3, sharex=True, dpi=150, figsize=[7, 5])
	# plt.suptitle('{} Failure Mode During {}, Total Displacement'.format(valvegroup, movement_type), y=0.95, fontsize=12)
	# df.plot(ax=ax[0],
	# 		x='Time (s)', y='tt',
	# 		marker='o',
	# 		markersize=2,
	# 		linestyle='None',
	# 		label='Measured')
	# x_filt.plot(ax=ax[0], x='Time (s)', y='XY Position', color ='#ff7f0e', label='Estimated Position')
	# x_filt.plot(ax=ax[1], x='Time (s)', y='XY Velocity', color ='#ff7f0e', label='Estimated Velocity')
	# x_filt.plot(ax=ax[2], x='Time (s)', y='XY Acceleration', color ='#ff7f0e', label='Estimated Acceleration')
	# ax[0].legend(loc='upper left')
	# ax[1].legend(loc='upper right')
	# ax[2].legend(loc='upper right')
	# ax[0].set_ylabel(r'$d$ (m)', color='#413839', fontsize=10)
	# ax[1].set_ylabel(r'$\dot{d}$ (m/s)', color='#413839', fontsize=10)
	# ax[2].set_ylabel(r'$\ddot{d}$ (m/$s^2$)', color='#413839', fontsize=10)


	# # Plot X vs Y
	# fig5, ax = plt.subplots(dpi=150, figsize=[9, 5])
	# plt.suptitle('{} Failure Mode During {}, Position Tracking'.format(valvegroup, movement_type), y=0.98, fontsize=12)
	# plt.title('Total Time: {} sec, Framerate: {} fps'.format(total_time, framerate), fontsize=12)
	# # plt.title('Measured Position with Kalman Estimate\nTotal Time: {} sec'.format(np.round(x_filt['Time (s)'].iloc[-1],1)))
	# df.plot(ax=ax,
	# 		x='t1', y='t2',
	# 		marker='o',
	# 		markersize=2,
	# 		linestyle='None',
	# 		label='Measured')
	# x_filt.plot(ax=ax, x='X Position', y='Y Position', label='Kalman Estimate')
	# ax.legend(loc='best')
	# plt.axis('equal')
	# plt.xlabel('X (m)')
	# plt.ylabel('Y (m)')
	# plt.xlim(left=-0.762, right=0.762)
	# plt.ylim(top=0.457, bottom=-0.457)

	
	# Plot the things
	# plt.show()


	# Return data (if applicable)
	return df, x_filt

	# covariance = pd.DataFrame(cov)
	# covariance['Time (s)'] = df['Time (s)']
	# covariance.plot(x='Time (s)')
	# plt.show()

if __name__ == "__main__":
	try:
		datafile = sys.argv[1]  # Specify data file
	except:
		pass  # Use defaults

	estimate_pose(datafile)