import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from filterpy.kalman import KalmanFilter
import sys

def estimate_pose(datafile):
	df = pd.read_csv(datafile)											# Columns should already be specified as ['Time (s)', 'r1', 'r2', 'r3', 't1', 't2', 't3'] when the datafile is written.
	# df.columns = ['Time (s)', 'r1', 'r2', 'r3', 't1', 't2', 't3'] 
	# df['t1'] = [-x for x in df['t1']]  # Invert the X values
	df['t2'] = [-x for x in df['t2']]  # Invert the Y values			# Because opencv returns the translation FROM the Aruco marker TO the camera, but we want to see it from the other side.
	df['dt'] = df['Time (s)'].diff().fillna(0)							# Add a column of time differentials (dt) between measurements
	zs = df[['t1', 't2']].values										# The actual measurements I made

	# n_iter = len(df)

	# Q = np.array([[9.0E-6, 0, 0], [0, 9.0E-6, 0], [0, 0, 9.0E-6]])	# Process variance/covariance. This is different than State Covariance. Should be of size NxN for N tracked states.
	Q = np.eye(6)*9.0E-6

	R = np.array([[9.0E-6, 	0		],									# Measurement variance/covariance. Should be size MxM for M measured states. Each value is the variance/covariance of the state measurement.
				  [0, 		9.0E-6 	]])	

	H = np.array([[1., 	0, 	0, 	0, 	0, 	0	],							# Measurement-space conversion. Should be size MxN for M measured states and N tracked states.
				  [0, 	0, 	0, 	1, 	0, 	0	]])							# Measured states are associated with a 1 (or whatever conversion factor is necessary to translate the state into an associated measurement). Everything else is 0.
																		# In this case, we're measuring position but not velocity or acceleration, so [1, 0, 0] such that Hx yields only position predictions in the residual.

	# P = np.array([[9.0E-6, 0, 0], [0, 9.0E-6, 0], [0, 0, 9.0E-6]])	# Initial state covariance (Expected variance of each state variable, including any covariances). Should be of size NxN for N tracked states.
																		# Diagonals of the covariance matrix contains the variance of each variable, and the off-diagonal elements contains the covariances.
	P = np.eye(6)*9.0E-6

	# State transition matrix of the system, to be updated every time the function is called because dt is not constant
	def state_transition_matrix(k=None):
		dt = df['dt'][k]
		# State transition model. Should be of size NxN for N tracked states.
		F = np.array([[1, 	dt,		0.5*dt**2, 	0, 		0, 		0			],	# x
					  [0, 	1, 		dt, 		0, 		0, 		0			],	# xdot
					  [0, 	0, 		1, 			0, 		0, 		0			], 	# xdotdot
					  [0, 	0, 		0, 			1, 		dt, 	0.5*dt**2	], 	# y
					  [0, 	0, 		0, 			0, 		1, 		dt			], 	# ydot
					  [0, 	0, 		0, 			0, 		0, 		1			]]) # ydotdot
		return F


	# B = np.array([0, 0, 0.02])
	# B = 0									# Optional control input model
	# u = 0									# Optional control inputs


	# Initial state. If I don't know what the initial state is, just set it to the first measurement.
	x = np.array([zs[0][0], 	# x
				  0, 			# xdot
				  0, 			# xdotdot
				  zs[0][1], 	# y
				  0, 			# ydot
				  0			])	# ydotdot

	kf = KalmanFilter(dim_x=6, dim_z=2, dim_u=0) 	# Initialize a KalmanFilter object.
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

	x_filt = pd.DataFrame(xs)						# Make a Pandas DataFrame from the Kalman-estimated states.
	# f, ax = plt.subplots(3, sharex=True)
	x_filt.columns = ['X Position', 'X Velocity', 'X Acceleration', 'Y Position', 'Y Velocity', 'Y Acceleration']
	x_filt['Time (s)'] = df['Time (s)']
	# x_filt.plot(ax=ax[0], x='Time (s)', y='Position')
	# x_filt.plot(ax=ax[1], x='Time (s)', y='Velocity')
	# x_filt.plot(ax=ax[2], x='Time (s)', y='Acceleration')
	# plt.show()

	dpi = 150
	f, ax = plt.subplots(dpi=dpi, figsize=[1280/dpi, 720/dpi])
	df.plot(x='t1', y='t2',
			marker='o',
			markersize=2,
			linestyle='None',
			ax=ax,
			label='Measured')
	x_filt.plot(x='X Position', y='Y Position', ax=ax, label='Kalman Estimate')
	ax.legend(loc='best')
	plt.axis('equal')
	# plt.xlim(right=0)
	# plt.ylim(top=0.9)
	plt.show()

	# covariance = pd.DataFrame(cov)
	# covariance['Time (s)'] = df['Time (s)']
	# covariance.plot(x='Time (s)')
	# plt.show()

if __name__ == "__main__":
	datafile = sys.argv[1]  # Specify data file
	estimate_pose(datafile)