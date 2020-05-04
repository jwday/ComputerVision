import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from filterpy.kalman import KalmanFilter

df = pd.read_csv('datafile.csv', delim_whitespace=False)
df.columns = ['Time (s)', 'r1', 'r2', 'r3', 't1', 't2', 't3']
df['dt'] = df['Time (s)'].diff().fillna(0)
zs = df['t1'].values					# The actual measurements I made

n_iter = len(df)

# Finite diff approx
vel_fd = []
acc_fd = []
for index, x in df['t1'].iloc[:-1].items():
	vel_fd.append((df['t1'].iloc[index+1] - x)/df['dt'].iloc[index+1])

for index, v in enumerate(vel_fd[:-1]):
	acc_fd.append((vel_fd[index+1] - v)/df['dt'].iloc[index+1])


# Kalman filter
Q = np.array([9.0E-6])					# Process variance/covariance. This is different than State Covariance.
R = np.array([5.0E-5])					# Measurement variance/covariance. Should be size NxN for N measured states. Each value is the variance of the state measurement
H = np.array([[1., 0, 0]])				# Measuring position, but not velocity or acceleration, so [1, 0, 0] such that Hx yields only position predictions in the residual
P = np.zeros(3)							# Initial State Covariance (Expected variance of each state variable, including any covariances).
										# Diagonals of the covariance matrix contains the variance of each variable, and the off-diagonal elements contains the covariances.

def state_transition_matrix(k=None):	# State transition matrix of the system, to be updated every time the function is called because dt is not constant
	dt = df['dt'][k]
	F = np.array([[1, dt, 0.5*dt**2], [0, 1, dt], [0, 0, 1]])
	return F

B = np.array([0, 0, 0.02])

# B = 0									# Optional control input model
# u = 0									# Optional control inputs
K = np.zeros(3)

# Initial x. If I don't know what the initial state is, just set it to my first measurement.
x = np.array([zs[0], 0, 0])

# def pos_vel_filter(x, H, P, R, Q, dt):
kf = KalmanFilter(dim_x=3, dim_z=1, dim_u=1) # Returns a KalmanFilter object which implements a constant acceleration model for a state [x dx ddx]T.
kf.x = x 							# Estimated position, velocity, and acceleration 
kf.F = state_transition_matrix(k=0) # State transition matrix
kf.H = H					   		# Measurement function
kf.R *= R                     		# Measurement uncertainty
kf.P[:] = P               			# [:] makes deep copy
kf.Q[:] = Q

# kf = pos_vel_filter(x, H=H, R=R, P=P, Q=Q, dt=0.12)

# # run the kalman filter and store the results
xs, cov = [], []
for i, z in enumerate(zs):
	kf.F = state_transition_matrix(i)
	kf.predict(u=0, B=B)
	kf.update(z)
	xs.append(kf.x)
	cov.append(kf.P.flatten())

x_filt = pd.DataFrame(xs)

f0, (ax0, ax1, ax2) = plt.subplots(3, sharex=True, dpi=300, figsize=[6, 5])
plt.suptitle('Finite Diff Approx (x-axis)', y=0.95, fontsize=14)
# x_filt.columns = ['Position', 'Velocity', 'Acceleration']
# x_filt['Time (s)'] = df['Time (s)']
ax0.plot(df['Time (s)'].values, df['t1'].values, label='Position')
ax0.set(ylabel='m')
ax0.legend()
ax1.plot(df['Time (s)'].iloc[:-1].values, vel_fd, label='Velocity')
ax1.set(ylabel='m/s')
ax1.legend()
ax2.plot(df['Time (s)'].iloc[:-2].values, acc_fd, label='Acceleration')
ax2.set(ylabel='$m/s^2$')
ax2.legend()
plt.xlabel('Time (s)')
# plt.show()



f, ax1 = plt.subplots(3, sharex=True, dpi=300, figsize=[6, 5])
plt.suptitle('Kalman-Estimated States (x-axis)', y=0.95, fontsize=14)
x_filt.columns = ['Position', 'Velocity', 'Acceleration']
x_filt['Time (s)'] = df['Time (s)']
x_filt.plot(ax=ax1[0], x='Time (s)', y='Position')
x_filt.plot(ax=ax1[1], x='Time (s)', y='Velocity')
x_filt.plot(ax=ax1[2], x='Time (s)', y='Acceleration')
ylabels = ['m', 'm/s', '$m/s^2$']
for i, ax1 in enumerate(ax1.flat):
    ax1.set(ylabel=ylabels[i])


plt.show()

f, ax2 = plt.subplots()
plt.title('Kalman-Estimated Position')
df.plot(x='Time (s)', y='t1', ax=ax2)
x_filt.plot(x='Time (s)', y='Position', ax=ax2)
plt.show()

covariance = pd.DataFrame(cov)
covariance['Time (s)'] = df['Time (s)']
covariance.plot(x='Time (s)')
plt.show()
