from kalmanThatBeezy import estimate_pose
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import linalg
from mpl_toolkits.mplot3d import Axes3D
from filterpy.kalman import KalmanFilter
import os
import sys
from pathlib import Path
import seaborn as sns
sns.axes_style("white")
# sns.set_style("white", {"xtick.major.size": 0, "ytick.major.size": 0, 'axes.grid': True, 'grid.color': '.8', 'grid.linestyle': '--',})
sns.set_style("whitegrid", {"xtick.major.size": 0, "ytick.major.size": 0, 'grid.linestyle': '--'})
sns.set_context("paper", font_scale = 1, rc={"grid.linewidth": .5})

directory = '~/Research Files/Videos/Discharges and Modes/A1+A2/'

def batch_kalman(directory):
	datafiles = [x.name for x in Path(directory).expanduser().glob('*.csv')]
	all_data = pd.DataFrame()

	for trial in datafiles:	
		valvegroup = trial.split('_')[0]
		movement_type = trial.split('_')[1]
		trial_number = trial.split('_')[2].split('.')[0]
		trial_name = movement_type + ' ' + trial_number

		if valvegroup == 'Pure':
			plot_title = 'Pure {} (No Failure Modes)'.format(movement_type)
		else:
			plot_title = '{} Failure Mode During {} Maneuver'.format(valvegroup, movement_type)

		meas, x_filt = estimate_pose(directory+trial, delim_whitespace=False)		# Returns the measured data and Kalman state esimates for the given trial

		trial_data = pd.concat([meas, x_filt, pd.DataFrame([trial_number for x in meas.index], columns=['Trial'])], axis=1)
		all_data = all_data.append(trial_data, ignore_index=True)

	# fig1, axs = plt.subplots(3, sharex=True, dpi=150, figsize=[7, 5])
	# fig1.suptitle(plot_title + ' (X-axis)', y=0.96, fontsize=12)
	# axs[0].set_title('(X-axis motion)', fontsize=10, color='dimgrey')
	# sns.lineplot(ax=axs[0], x="Time (s)", y="X Position", hue='Trial', data=all_data, legend='full')
	# sns.lineplot(ax=axs[1], x="Time (s)", y="X Velocity", hue='Trial', data=all_data, legend='full')
	# sns.lineplot(ax=axs[2], x="Time (s)", y="X Acceleration", hue='Trial', data=all_data, legend='full')
	# axs[0].set_ylabel(r'$d_x$ (m)', fontsize=10)
	# axs[1].set_ylabel(r'$\dot{d_x}$ (m/s)', fontsize=10)
	# axs[2].set_ylabel(r'$\ddot{d_x}$ (m/$s^2$)', fontsize=10)
	# plt.xlim(left=0, right=14)
	# for i, x in enumerate(axs):
	# 	axs[i].legend(loc='right')
	# plt.savefig('x.png')

	# fig2, axs = plt.subplots(3, sharex=True, dpi=150, figsize=[7, 5])
	# fig2.suptitle(plot_title , y=0.96, fontsize=12)
	# axs[0].set_title('(Y-axis motion)', fontsize=10, color='dimgrey')
	# sns.lineplot(ax=axs[0], x="Time (s)", y="Y Position", hue='Trial', data=all_data, legend='full')
	# sns.lineplot(ax=axs[1], x="Time (s)", y="Y Velocity", hue='Trial', data=all_data, legend='full')
	# sns.lineplot(ax=axs[2], x="Time (s)", y="Y Acceleration", hue='Trial', data=all_data, legend='full')
	# axs[0].set_ylabel(r'$d_y$ (m)', fontsize=10)
	# axs[1].set_ylabel(r'$\dot{d_y}$ (m/s)', fontsize=10)
	# axs[2].set_ylabel(r'$\ddot{d_y}$ (m/$s^2$)', fontsize=10)
	# plt.xlim(left=0, right=14)
	# for i, x in enumerate(axs):
	# 	axs[i].legend(loc='right')
	# plt.savefig('y.png')

	# fig3, axs = plt.subplots(3, sharex=True, dpi=150, figsize=[7, 5])
	# fig3.suptitle(plot_title, y=0.97, fontsize=14)
	# axs[0].set_title('(Total translational motion)', fontsize=10, color='dimgrey')
	# sns.lineplot(ax=axs[0], x="Time (s)", y="XY Position", hue='Trial', data=all_data, legend='full')
	# sns.lineplot(ax=axs[1], x="Time (s)", y="XY Velocity", hue='Trial', data=all_data, legend='full')
	# sns.lineplot(ax=axs[2], x="Time (s)", y="XY Acceleration", hue='Trial', data=all_data, legend='full')
	# axs[0].set_ylabel(r'$d$ (m)', fontsize=10)
	# axs[1].set_ylabel(r'$\dot{d}$ (m/s)', fontsize=10)
	# axs[2].set_ylabel(r'$\ddot{d}$ (m/$s^2$)', fontsize=10)
	# plt.xlim(left=0, right=14)
	# for i, x in enumerate(axs):
	# 	axs[i].legend(loc='right')
	# plt.savefig('xy.png')

	# fig4, axs = plt.subplots(3, sharex=True, dpi=150, figsize=[7, 5])
	# fig4.suptitle(plot_title, y=0.96, fontsize=12)
	# axs[0].set_title('(Rotational motion)', fontsize=10, color='dimgrey')
	# sns.lineplot(ax=axs[0], x="Time (s)", y="r1 Angle", hue='Trial', data=all_data, legend='full')
	# sns.lineplot(ax=axs[1], x="Time (s)", y="r1 Velocity", hue='Trial', data=all_data, legend='full')
	# sns.lineplot(ax=axs[2], x="Time (s)", y="r1 Acceleration", hue='Trial', data=all_data, legend='full')
	# axs[0].set_ylabel(r'$\Theta$ (deg)', fontsize=10)
	# axs[1].set_ylabel(r'$\omega$ (deg/s)', fontsize=10)
	# axs[2].set_ylabel(r'$\alpha$ (deg/$s^2$)', fontsize=10)
	# plt.xlim(left=0, right=14)
	# for i, x in enumerate(axs):
	# 	axs[i].legend(loc='right')
	# plt.savefig('r.png')



	fig5, axs = plt.subplots(3, 2, sharex=True, dpi=300, figsize=[6, 4.5])
	fig5.suptitle(plot_title, y=0.98, fontsize=12)

	axs[0, 0].set_title('Total Translational Motion', fontsize=9, color='dimgrey', y=1.04)
	axs[0, 1].set_title('Rotational Motion', fontsize=9, color='dimgrey', y=1.04)

	sns.lineplot(ax=axs[0, 0], x="Time (s)", y="XY Position", hue='Trial', data=all_data, legend='full')
	sns.lineplot(ax=axs[1, 0], x="Time (s)", y="XY Velocity", hue='Trial', data=all_data, legend='full')
	sns.lineplot(ax=axs[2, 0], x="Time (s)", y="XY Acceleration", hue='Trial', data=all_data, legend='full')
	sns.lineplot(ax=axs[0, 1], x="Time (s)", y="r1 Angle", hue='Trial', data=all_data, legend='full')
	sns.lineplot(ax=axs[1, 1], x="Time (s)", y="r1 Velocity", hue='Trial', data=all_data, legend='full')
	sns.lineplot(ax=axs[2, 1], x="Time (s)", y="r1 Acceleration", hue='Trial', data=all_data, legend='full')

	axs[0, 0].set_ylabel(r'$d$ (m)', fontsize=8)
	axs[1, 0].set_ylabel(r'$\dot{d}$ (m/s)', fontsize=8)
	axs[2, 0].set_ylabel(r'$\ddot{d}$ (m/$s^2$)', fontsize=8)
	axs[0, 1].set_ylabel(r'$\Theta$ (deg)', fontsize=8)
	axs[1, 1].set_ylabel(r'$\omega$ (deg/s)', fontsize=8)
	axs[2, 1].set_ylabel(r'$\alpha$ (deg/$s^2$)', fontsize=8)
	fig5.align_ylabels()

	class ScalarFormatterForceFormat(mpl.ticker.ScalarFormatter):
		def _set_format(self):  # Override function that finds format to use.
			self.format = "%1.1f"  # Give format here

	plt.xlim(left=0, right=14)

	for ax in axs.flat:
			ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
			
			yfmt = ScalarFormatterForceFormat()
			yfmt.set_powerlimits((0,0))
			ax.yaxis.set_major_formatter(yfmt)
			ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useMathText=True)
			ax.tick_params(axis='y', labelsize=6, pad=0)
			ax.yaxis.offsetText.set_fontsize(6)

			ax.tick_params(axis='x', labelsize=6, pad=0)
			ax.xaxis.label.set_size(8)
			ax.set(xlabel=r'Time $(sec)$')

	plt.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.1, wspace=0.3, hspace=0.2)
	plt.savefig(directory + 'all.png')
	plt.show()

if __name__ == "__main__":
	try:
		directory = sys.argv[1]  # Specify directory containing data files you want to analyze together
	except:
		pass  # Use defaults

	batch_kalman(directory)