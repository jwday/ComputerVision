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

datafile = './datafile.csv'

def estimate_pose(datafile, delim_whitespace=False):
	df = pd.read_csv(datafile)											# Columns should already be specified as ['Time (s)', 'r1', 'r2', 'r3', 't1', 't2', 't3'] when the datafile is written.
	df.columns = ['Time (s)', 'r1', 'r2', 'r3', 't1', 't2', 't3']
	df['t2'] = [-x for x in df['t2']]									# Because opencv returns the translation FROM the Aruco marker TO the camera, but we want to see it from the other side.
	df['dt'] = df['Time (s)'].diff()									# Add a column of time differentials (dt) between measurements
	df['dt'] = df['dt'].fillna(df['dt'].mean())							# Fill any NA's with the average dt

	zs = df[['t1', 't2', 'r1', 'r2']].values									# Create a Numpy array of the measurements for use in the Kalman filter


	# -----------------------------------------------------------------
	# STATE COVARIANCE MATRIX (P)
	# -----------------------------------------------------------------
	# P = np.array([[9.0E-6, 0, 0], [0, 9.0E-6, 0], [0, 0, 9.0E-6]])	# Initial state covariance (expected variance of each state variable). Should be of size NxN for N tracked states.
	# P = np.eye(6)*9.0E-6
	P = np.array([[9.0E-4,		 0,		 0,		 0,		 0,		 0,		 0,		 0,		 0,		0,		0,		0],  # x
				  [		0,	9.0E-2,		 0,		 0,		 0,		 0,		 0,		 0,		 0,		0,		0,		0],  # xdot
				  [		0,		 0,	9.0E-0,		 0,		 0,		 0,		 0,		 0,		 0,		0,		0,		0],  # xdotdot
				  [		0,		 0,		 0,	9.0E-4,		 0,		 0,		 0,		 0,		 0,		0,		0,		0],  # y
				  [		0,		 0,		 0,		 0,	9.0E-2,		 0,		 0,		 0,		 0,		0,		0,		0],  # ydot
				  [		0,		 0,		 0,		 0,		 0,	9.0E-0,		 0,		 0,		 0,		0,		0,		0],  # ydotdot
				  [		0,		 0,		 0,		 0,		 0,		 0,		 1,		 0,		 0,		0,		0,		0],  # r1
				  [		0,		 0,		 0,		 0,		 0,		 0,		 0,	   100,		 0,		0,		0,		0],  # r1dot
				  [		0,		 0,		 0,		 0,		 0,		 0,		 0,		 0,	 10000,		0,		0,		0],  # r1dotdot
				  [		0,		 0,		 0,		 0,		 0,		 0,		 0,		 0,		 0,		1,		0,		0],  # r2
				  [		0,		 0,		 0,		 0,		 0,		 0,		 0,		 0,		 0,		0,	  100,		0],  # r2dot
				  [		0,		 0,		 0,		 0,		 0,		 0,		 0,		 0,		 0,		0,		0,	10000]]) # r2dotdot


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
	Q_r1 = Q_discrete_white_noise(dim=3, dt=df['dt'].mean(), var=0.01, block_size=1)	# Different white noise for Theta
	Q_r2 = Q_r1
	Q = linalg.block_diag(Q_t1, Q_t2, Q_r1, Q_r2)

	# -----------------------------------------------------------------
	# MEASUREMENT COVARIANCE MATRIX (R)
	# -----------------------------------------------------------------
	R = np.array([[1.0E-3, 		 0,			0,			0],				# Measurement variance/covariance. Should be size MxM for M measured states. Each value is the variance/covariance of the state measurement.
				  [	 0,		1.0E-3,		 	0,			0],
				  [	 0,   		 0, 	100.0,			0],
				  [	 0,   		 0, 		0,		100.0]])


	# -----------------------------------------------------------------
	# STATE-SPACE TO MEASUREMENT-SPACE CONVERSION MATRIX (H)
	# -----------------------------------------------------------------
	H = np.array([[1., 	0, 	0, 	0, 	0, 	0,	0,	0,	0,	0,	0,	0],					# Measurement-space conversion. Should be size MxN for M measured states and N tracked states.
				  [ 0, 	0, 	0, 	1, 	0, 	0,	0,	0,	0,	0,	0,	0],					# Measured states are associated with a 1 (or whatever conversion factor is necessary to translate the state into an associated measurement). Everything else is 0.
				  [ 0, 	0, 	0, 	0, 	0, 	0,	1,	0,	0,	0,	0,	0],
				  [ 0, 	0, 	0, 	0, 	0, 	0,	0,	0,	0,	1,	0,	0]])				# In this case, we're measuring position but not velocity or acceleration, so [1, 0, 0] such that Hx yields only position predictions in the residual.


	# -----------------------------------------------------------------
	# STATE TRANSITION MODEL MATRIX (F)
	# -----------------------------------------------------------------
	def state_transition_matrix(k=None):								# State transition matrix of the system, to be updated every time the function is called because dt is not constant. Should be of size NxN for N tracked states.
		dt = df['dt'][k]
		F = np.array([[1, 	dt,	0.5*dt**2, 		0, 		0,		   0,		0,		0,			0,		0,		0,			0],	 # x
					  [0, 	 1, 	   dt, 		0, 		0,		   0,		0,		0,			0,		0,		0,			0],	 # xdot
					  [0, 	 0, 		1, 		0, 		0,		   0,		0,		0,			0,		0,		0,			0],  # xdotdot
					  [0, 	 0, 		0, 		1, 	   dt, 0.5*dt**2,		0,		0,			0,		0,		0,			0],  # y
					  [0, 	 0, 		0, 		0, 		1, 		  dt,		0,		0,			0,		0,		0,			0],  # ydot
					  [0, 	 0, 		0, 		0, 		0, 		   1,		0,		0,			0,		0,		0,			0],	 # ydotdot
					  [0, 	 0, 		0, 		0, 		0, 		   0,		1,		dt,	0.5*dt**2,		0,		0,			0],	 # r1
					  [0, 	 0, 		0, 		0, 		0, 		   0,		0,		1,		   dt,		0,		0,			0],	 # r1dot
					  [0, 	 0, 		0, 		0, 		0, 		   0,		0,		0,			1,		0,		0,			0],	 # r1dotdot
					  [0, 	 0, 		0, 		0, 		0, 		   0,		0,		0,			0,		1,		dt,	0.5*dt**2],  # r2
					  [0, 	 0, 		0, 		0, 		0, 		   0,		0,		0,			0,		0,		1,		   dt],  # r2dot
					  [0, 	 0, 		0, 		0, 		0, 		   0,		0,		0,			0,		0,		0,			1]]) # r2dotdot
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
	x = np.array([zs[0][0],  # x
				  		 0,  # xdot
				  		 0,  # xdotdot
				  zs[0][1],  # y
				  		 0,  # ydot
				  		 0,  # ydotdot
				  zs[0][2],  # r1
				  		 0,  # r1dot
				  		 0,  # r1dotdot
				  zs[0][3],	 # r2
						 0,	 # r2dot
						 0]) # r2dotdot


	# -----------------------------------------------------------------
	# KALMAN FILTER
	# -----------------------------------------------------------------
	# Set up the Kalman Filter object
	kf = KalmanFilter(dim_x=12, dim_z=4, dim_u=0) 	# Initialize a KalmanFilter object.
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
	x_filt.columns = ['X Position', 'X Velocity', 'X Acceleration', 'Y Position', 'Y Velocity', 'Y Acceleration', 'r1', 'r1 Velocity', 'r1 Acceleration', 'r2', 'r2 Velocity', 'r2 Acceleration']
	x_filt['Time (s)'] = df['Time (s)']
	total_time = np.round(x_filt['Time (s)'].iloc[-1],1)
	framerate = np.round(df.count()[0]/x_filt['Time (s)'].iloc[-1], 2)


	# -----------------------------------------------------------------
	# PLOT RESULTS
	# -----------------------------------------------------------------
	# Plot just X
	fig1, ax = plt.subplots(3, sharex=True, dpi=150, figsize=[6, 5])
	plt.suptitle('Kalman-Estimated States (x-axis)', y=0.95, fontsize=14)
	df.plot(ax=ax[0],
			x='Time (s)', y='t1',
			marker='o',
			markersize=2,
			linestyle='None',
			label='Measured')
	x_filt.plot(ax=ax[0], x='Time (s)', y='X Position')
	x_filt.plot(ax=ax[1], x='Time (s)', y='X Velocity')
	x_filt.plot(ax=ax[2], x='Time (s)', y='X Acceleration')
	ax[0].legend()

	# Plot just Y
	fig2, ax = plt.subplots(3, sharex=True, dpi=150, figsize=[6, 5])
	plt.suptitle('Kalman-Estimated States (y-axis)', y=0.95, fontsize=14)
	df.plot(ax=ax[0],
			x='Time (s)', y='t2',
			marker='o',
			markersize=2,
			linestyle='None',
			label='r1 Measured')
	x_filt.plot(ax=ax[0], x='Time (s)', y='Y Position')
	x_filt.plot(ax=ax[1], x='Time (s)', y='Y Velocity')
	x_filt.plot(ax=ax[2], x='Time (s)', y='Y Acceleration')
	ax[0].legend()

	# Plot just Theta
	fig3, ax = plt.subplots(3, sharex=True, dpi=150, figsize=[6, 5])
	plt.suptitle('Kalman-Estimated States (Theta)', y=0.95, fontsize=14)
	df.plot(ax=ax[0],
			x='Time (s)', y='r1',
			marker='o',
			markersize=2,
			linestyle='None',
			label='Measured')
	df.plot(ax=ax[1],
			x='Time (s)', y='r2',
			marker='o',
			markersize=2,
			linestyle='None',
			label='r1 Measured')
	x_filt.plot(ax=ax[0], x='Time (s)', y='r1')
	x_filt.plot(ax=ax[1], x='Time (s)', y='r2')
	# x_filt.plot(ax=ax[2], x='Time (s)', y='Angular Acceleration')
	ax[0].legend()
	ax[1].legend()


	# Plot X vs Y
	fig4, ax = plt.subplots(dpi=150, figsize=[9, 5])
	plt.suptitle('Measured Position with Kalman Estimate', y=0.98, fontsize=16)
	plt.title('Total Time: {} sec, Framerate: {} fps'.format(total_time, framerate), fontsize=12)
	# plt.title('Measured Position with Kalman Estimate\nTotal Time: {} sec'.format(np.round(x_filt['Time (s)'].iloc[-1],1)))
	df.plot(ax=ax,
			x='t1', y='t2',
			marker='o',
			markersize=2,
			linestyle='None',
			label='Measured')
	x_filt.plot(ax=ax, x='X Position', y='Y Position', label='Kalman Estimate')
	ax.legend(loc='best')
	plt.axis('equal')
	plt.xlabel('X (m)')
	plt.ylabel('Y (m)')
	# plt.xlim(left=-1.778, right=0)
	# plt.ylim(top=1, bottom=0)
	
	plt.show()





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