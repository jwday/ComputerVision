import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('datafile.csv', delim_whitespace=True)
df.columns = ['Time (s)', 'r1', 'r2', 'r3', 't1', 't2', 't3']

n_iter = len(df)
sz = (n_iter, 2)

# x = 0
# z = np.rand.normal(x, 0.1, size=sz)
z = df[['t1', 't2']].values

q = 1  # Process variance (How steady is your process?)
r = 1  # Measurement variance (How noisy is your measurement?)
# The relative value between these two values is what matters, not their absolute values
# If there's interaction, there could be cross terms, so q and r would be matrices to model that crossover

x_hat = np.zeros(sz)  # x_hat is the posteri estimate of x (the current estimate of x, after i measureme it at this time and my q and r)
p = np.zeros(sz)  # p is the posteri error estimate (the current estimate of how wrong it is)
x_hat_minus = np.zeros(sz)  # x_hat_minus is the priori estimate of x (my best guess of where x will be, before I measure it, based on my previous measurement and guess)
p_minus = np.zeros(sz) # p_minus is the estimate of error (estimate of how wrong i used to be)
K = np.zeros(sz)  # K is the gain, or 'blending factor'

x_hat[0] = z[0]  # Initial guess for x (when set to z[0], our 'best guess' for the first measurement is the first data point)
p[0] = 1  # Initial guess for error p

for k in range(1, n_iter):
    x_hat_minus[k] = x_hat[k-1]
    p_minus[k] = p[k-1] + q

    K[k] = p_minus[k] / (p_minus[k] + r)
    x_hat[k] = x_hat_minus[k] + K[k]*(z[k] - x_hat_minus[k])  # New estimate of x after I measured it: previous estimate + blending factor*(measured - estimated)
    p[k] = (1 - K[k])*p_minus[k]

# plt.figure()
# plt.plot(z, 'k+', label='measurements')
# plt.plot(x_hat, 'b-', label='kalman estimate')
# plt.legend()
# plt.show()

plt.figure()
plt.plot(z[:,0], z[:,1], 'k+', label='measurements')
plt.plot(x_hat[:,0], x_hat[:,1], 'b-', label='kalman estimate')
plt.legend()
plt.show()

plt.figure()
plt.plot(p[:, 0], 'blue', label='t1 error est.')
plt.plot(p[:, 1], 'black', label='t2 error est.')
plt.legend()
plt.show()