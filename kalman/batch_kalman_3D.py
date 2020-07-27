from kalmanThatBeezy import estimate_pose
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from mpl_toolkits.mplot3d import Axes3D
from filterpy.kalman import KalmanFilter
import os
import sys
from pathlib import Path
import seaborn as sns
sns.set()

directory = '~/Research Files/Videos/Discharges and Modes/A2+D3/'
datafiles = [x.name for x in Path(directory).expanduser().glob('*.csv')]
all_data = pd.DataFrame()

for trial in datafiles:	
	valvegroup = trial.split('_')[0]
	movement_type = trial.split('_')[1]
	trial_number = trial.split('_')[2].split('.')[0]
	trial_name = movement_type + ' ' + trial_number

	meas, x_filt = estimate_pose(directory+trial, delim_whitespace=False)		# Returns the measured data and Kalman state esimates for the given trial

	trial_data = pd.concat([meas, x_filt, pd.DataFrame([trial_name for x in meas.index], columns=['trial name'])], axis=1)
	all_data = all_data.append(trial_data, ignore_index=True)

fig1, axs = plt.subplots(3, sharex=True, dpi=150, figsize=[7, 5])
fig1.suptitle('{} Failure Mode During {} (X-axis)'.format(valvegroup, movement_type), y=0.97, fontsize=12)
sns.lineplot(ax=axs[0], x="Time (s)", y="X Position", hue='trial name', data=all_data)
sns.lineplot(ax=axs[1], x="Time (s)", y="X Velocity", hue='trial name', data=all_data)
sns.lineplot(ax=axs[2], x="Time (s)", y="X Acceleration", hue='trial name', data=all_data)

fig2, axs = plt.subplots(3, sharex=True, dpi=150, figsize=[7, 5])
fig2.suptitle('{} Failure Mode During {} (Y-axis)'.format(valvegroup, movement_type), y=0.97, fontsize=12)
sns.lineplot(ax=axs[0], x="Time (s)", y="Y Position", hue='trial name', data=all_data)
sns.lineplot(ax=axs[1], x="Time (s)", y="Y Velocity", hue='trial name', data=all_data)
sns.lineplot(ax=axs[2], x="Time (s)", y="Y Acceleration", hue='trial name', data=all_data)

fig3, axs = plt.subplots(3, sharex=True, dpi=150, figsize=[7, 5])
fig3.suptitle('{} Failure Mode During {} (Total Translation)'.format(valvegroup, movement_type), y=0.97, fontsize=12)
sns.lineplot(ax=axs[0], x="Time (s)", y="XY Position", hue='trial name', data=all_data)
sns.lineplot(ax=axs[1], x="Time (s)", y="XY Velocity", hue='trial name', data=all_data)
sns.lineplot(ax=axs[2], x="Time (s)", y="XY Acceleration", hue='trial name', data=all_data)

fig4, axs = plt.subplots(3, sharex=True, dpi=150, figsize=[7, 5])
fig4.suptitle('{} Failure Mode During {} (Rotation)'.format(valvegroup, movement_type), y=0.97, fontsize=12)
sns.lineplot(ax=axs[0], x="Time (s)", y="r1 Angle", hue='trial name', data=all_data)
sns.lineplot(ax=axs[1], x="Time (s)", y="r1 Velocity", hue='trial name', data=all_data)
sns.lineplot(ax=axs[2], x="Time (s)", y="r1 Acceleration", hue='trial name', data=all_data)

# plt.title('{} Failure Mode'.format(valvegroup))
plt.show()

if __name__ == "__main__":
	try:
		directory = sys.argv[1]  # Specify directory containing data files you want to analyze together
	except:
		pass  # Use defaults

	# estimate_pose(directory)