import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from filterpy.kalman import KalmanFilter

datafile = '/home/josh/ComputerVision/kalman/datafile_720p_5.0fps.csv'

df = pd.read_csv(datafile, delim_whitespace=False)
df.columns = ['Time (s)', 'r1', 'r2', 'r3', 't1', 't2', 't3']
df['dt'] = df['Time (s)'].diff().fillna(0)
zs = df['t1'].values					# The actual measurements I made
zs_noise = [i + np.random.normal(0,0.01,1)[0] for i in zs]
zs = zs_noise

n_iter = len(df)

# Finite diff approx
vel_fd = []
acc_fd = []
for index, x in df['t1'].iloc[:-1].items():
	vel_fd.append((df['t1'].iloc[index+1] - x)/df['dt'].iloc[index+1])

for index, v in enumerate(vel_fd[:-1]):
	acc_fd.append((vel_fd[index+1] - v)/df['dt'].iloc[index+1])


# Kalman filter
R = np.array([1E-2])					# Measurement variance/covariance. Should be size NxN for N measured states. Each value is the variance of the state measurement
H = np.array([[1., 0, 0]])				# Measuring position, but not velocity or acceleration, so [1, 0, 0] such that Hx yields only position predictions in the residual
P = np.array([1E-3, 1E-1, 10.])			# Initial State Covariance (Expected variance of each state variable, including any covariances).
										# Diagonals of the covariance matrix contains the variance of each variable, and the off-diagonal elements contains the covariances.
# P = np.array([[	 1E-3,		1E-2,	  1E-1],  	# x
# 			  [	 1E-2,	 	1E-1,		 1],  	# xdot
# 			  [	 1E-1,		   1,		1E1]]) 	# xdotdot

# Q = np.array([[	 1E-7,		 1E-6,		 1E-5],  	# x
# 			  [	 1E-6,	  	 1E-5,		 1E-4],  	# xdot
# 			  [	 1E-5,		 1E-4,		 1E-3]]) 	# xdotdot
# Q = np.array([[	 1E-3,		 1E-2,		  3E-2],  	# x
# 			  [	 1E-2,	  	 1E-1,		  3E-1],  	# xdot
# 			  [	 3E-2,		 3E-1,		  1E0]]) 	# xdotdot
			  
# Q = np.array([1E-2])					# Process variance/covariance. This is different than State Covariance., 
from filterpy.common import Q_discrete_white_noise
Q = Q_discrete_white_noise(dim=3, dt=df['dt'].mean(), var=1E-3, block_size=1)


def state_transition_matrix(k=None):	# State transition matrix of the system, to be updated every time the function is called because dt is not constant
	dt = df['dt'][k]
	F = np.array([[1, dt, 0.5*dt**2], [0, 1, dt], [0, 0, 1]])
	return F

B = np.array([0, 0, 0])

# B = 0									# Optional control input model
# u = 0									# Optional control inputs
K = np.zeros(3)

# Initial x. If I don't know what the initial state is, just set it to my first measurement.
x = np.array([zs[0], 0, 0])

# def pos_vel_filter(x, H, P, R, Q, dt):
kf = KalmanFilter(dim_x=3, dim_z=1, dim_u=0) # Returns a KalmanFilter object which implements a constant acceleration model for a state [x dx ddx]T.
kf.x = x 							# Estimated position, velocity, and acceleration 
kf.F = state_transition_matrix(k=0) # State transition matrix
kf.H = H					   		# Measurement function
kf.R *= R                     		# Measurement uncertainty
kf.P[:] = P               			# [:] makes deep copy
kf.Q[:] = Q

# kf = pos_vel_filter(x, H=H, R=R, P=P, Q=Q, dt=0.12)

# # run the kalman filter and store the results
xs, cov, foo = [], [], []
for i, z in enumerate(zs):
	kf.F = state_transition_matrix(i)
	kf.predict(u=0, B=B)
	kf.update(z)
	xs.append(kf.x)
	cov.append(kf.P.flatten())
	foo.append(kf.Q.flatten())

x_filt = pd.DataFrame(xs)

f0, (ax0, ax1, ax2) = plt.subplots(3, sharex=True, dpi=300, figsize=[6, 5])
plt.suptitle('Finite Diff Approx (x-axis)', y=0.95, fontsize=14)
# x_filt.columns = ['Position', 'Velocity', 'Acceleration']
# x_filt['Time (s)'] = df['Time (s)']
ax0.plot(df['Time (s)'].values, zs_noise, label='Position')
ax0.set(ylabel='m')
ax0.legend()
ax0.set_xlim([0, 82])
plt.legend(loc='upper left')
ax1.plot(df['Time (s)'].iloc[:-1].values, vel_fd, label='Velocity')
ax1.set(ylabel='m/s')
ax1.legend()
ax1.set_xlim([0, 82])
plt.legend(loc='upper left')
ax2.plot(df['Time (s)'].iloc[:-2].values, acc_fd, label='Acceleration')
ax2.set(ylabel='$m/s^2$')
ax2.legend()
ax2.set_xlim([0, 82])
plt.legend(loc='upper left')
plt.xlabel('Time (s)')
# plt.show()



f, ax1 = plt.subplots(3, sharex=True, dpi=300, figsize=[6, 5])
plt.suptitle('Kalman-Estimated States (x-axis)', y=0.95, fontsize=14)
x_filt.columns = ['Position', 'Velocity', 'Acceleration']
x_filt['Time (s)'] = df['Time (s)']
x_filt.plot(ax=ax1[0], x='Time (s)', y='Position')
ax1[0].set_xlim([0, 82])
x_filt.plot(ax=ax1[1], x='Time (s)', y='Velocity')
ax1[1].set_xlim([0, 82])
x_filt.plot(ax=ax1[2], x='Time (s)', y='Acceleration')
ax1[2].set_xlim([0, 82])
ylabels = ['m', 'm/s', '$m/s^2$']
for i, ax1 in enumerate(ax1.flat):
    ax1.set(ylabel=ylabels[i])


# plt.show()

f, ax2 = plt.subplots(dpi=300, figsize=[12, 4])
plt.title('Kalman-Estimated Position (x-axis)')
df.plot(x='Time (s)', y='t1', ax=ax2, label='Measured')
ax2.plot(df['Time (s)'].values, zs_noise, label='Noisy Data', linestyle='None', marker='o', markersize=4)
x_filt.plot(x='Time (s)', y='Position', ax=ax2, label='Estimated', linestyle='--')
plt.ylabel('X (m)')
ax2.legend()
plt.show()

covariance = pd.DataFrame(cov)
covariance['Time (s)'] = df['Time (s)']
covariance.plot(x='Time (s)')
# plt.show()
