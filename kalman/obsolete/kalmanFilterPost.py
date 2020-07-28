import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def derivative(data):	# 'data' should be a two-column array. First column time, second column data
	derivative_of_data = np.zeros((len(data) - 1, 2))	# Derivative data set has one fewer points (need two points to take derivative)
	
	for i, k in enumerate(derivative_of_data):
		derivative_of_data[i][0] = (data[i+1][0] + data[i][0]) / 2								# the derivative is assumed to be instantaneous at a time between the two data points
		derivative_of_data[i][1] = (data[i+1][1] - data[i][1]) / (data[i+1][0] - data[i][0])	# delta-x / delta-t

	return derivative_of_data


df = pd.read_csv('datafile.csv', delim_whitespace=True)
df.columns = ['Time (s)', 'r1', 'r2', 'r3', 't1', 't2', 't3']
z = df[['t1']].values	# The actual measurements I made

n_iter = len(df)
sz = (n_iter, z.shape[1])


Q = np.ones(z.shape[1])  	# Process error covariance (How steady is your process?)
R = np.ones(z.shape[1])		# Measurement error covariance (How noisy is your measurement?)
							# The relative value between these two values is what matters, not their absolute values.
							# If there's interaction between variables, then there will be cross terms.
H = np.ones(z.shape[1])		# Used to convert and combine the measured value with the predicted value. For multivariable filters, some variables are not measured--this will allow the filter to combine measured with predicted values
							# In the case of position & velocity, only position is measured. So we make the matrix [1, 0] such that y = z - Hx 

x_hat_minus = np.zeros(sz)  # The "a priori" estimate of x (my estimate of x before I measure it, based on my previous measurements and guesses).
x_hat = np.zeros(sz)  		# The "a posteriori" estimate of x (my "improved" estimate of x after I measure it (assuming my measurement is not perfect, i.e. noisy), based on a weighted combination of my original estimate and my measurement).

p_minus = np.zeros(sz) 		# The "a priori" estimate error covariance (my estimate of how wrong x is, compared to its actual value, before I measure it).
p = np.zeros(sz)  			# The "a posteriori" estimate error covariance (my estimate of how wrong x is, compared to its actual value, after I measure it).

K = np.zeros(sz)  			# K is the gain, or 'blending factor'.


# Initial values for starting the Kalman filter loop
x_hat[0] = z[0]  			# Initial guess for x (when set to z[0], our 'best guess' for the first measurement is the first data point).
p[0] = 1  					# Initial guess for error covariance.

for k in range(1, n_iter):
	# Predict
    x_hat_minus[k] = x_hat[k-1]		# The "a priori" estimate of x at k, before I measure it (it's really just the "a posteriori" estimate from the previous time step).
    p_minus[k] = p[k-1] + Q			# The "a priori" estimate error covariance at k, before I measure x at k.

	# Compute Gain
    K[k] = p_minus[k]*np.transpose(H) / (H*p_minus[k]*np.transpose(H) + R)		# Gain, or weighting parameter. K is chosen to minimize the "a posteriori" estimate error covariance (p). This is part of the derivation that's rooted in statistics.

	# Update 
    x_hat[k] = x_hat_minus[k] + K[k]*( z[k] - H*x_hat_minus[k] )  				# The "a posteriori" estimate of x at k (my improved estimate of x after I measured it).
    p[k] = (1 - K[k]*H)*p_minus[k]												# The "a posteriori" estimate error covariance at k (my improved estimate of the error between what I estimate x to be vs. what it actually is).


# px = df[['Time (s)', 't1']].values
# px_hat = np.zeros(sz)
# for k in range(1, n_iter):
# 	px_hat[k] = np.array([px[k][0], x_hat[k][0]])
# vx = derivative(px)
# vx_hat = derivative(px_hat)



# plt.figure()
# plt.plot(z, 'k+', label='measurements')
# plt.plot(x_hat, 'b-', label='kalman estimate')
# plt.legend()
# plt.show()

# plt.figure()
# plt.plot(z[:,0], z[:,1], 'k+', label='measurements')
# plt.plot(x_hat[:,0], x_hat[:,1], 'b-', label='kalman estimate')
# plt.legend()
# plt.show()

# plt.figure()
# plt.plot(p[:, 0], 'blue', label='t1 error est.')
# plt.plot(p[:, 1], 'black', label='t2 error est.')
# plt.legend()
# plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.plot(z[:,0], z[:,1], z[:,2], 'k+', label='measurements')
plt.plot(x_hat[:,0], x_hat[:,1], x_hat[:,2], 'b-', label='kalman estimate')
plt.legend()
plt.show()