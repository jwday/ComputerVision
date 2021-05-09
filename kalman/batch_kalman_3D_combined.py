from kalman3D import estimate_pose
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
from pathlib import Path
import seaborn as sns

sns.set()
sns.axes_style("white")
sns.set_style("whitegrid", {"xtick.major.size": 0, "ytick.major.size": 0, 'grid.linestyle': '--'})
sns.set_context("paper", font_scale = 1, rc={"grid.linewidth": .5})
sns.set_palette("colorblind")

# directory = '~/Research Files/Videos/Discharges and Modes/A1+A2/'
directories = [ '~/Research-Files/Videos/Discharges and Modes/Pure Rotation',
				'~/Research-Files/Videos/Discharges and Modes/A2/Rotation',
				'~/Research-Files/Videos/Discharges and Modes/A3/Rotation']
dev_run = False

# def batch_kalman(directory, dev_run):
# print('What is dev_run? It is: {}'.format(dev_run))

# if dev_run == True:
	# dpi = 150
	# print('dpi is 150')
# else:
# 	dpi = 300
	# print('dpi is 300')

# Creates a dict object with key : value pair of failure mode and list of datafile names based on whatever directories are listed in 'directories'
# i.e. {'A1+A2' : ['A1+A2_Translation_1.csv', 'A1+A2_Translation_2.csv', 'A1+A2_Translation_3.csv']}
all_trials = {}
for i, mode in enumerate([Path(x).expanduser().glob('*.csv') for x in directories]):
	mode_trials = [directories[i] + '/' + x.name for x in mode]
	mode_label = mode_trials[0].split('/')[-1].split('_')[0]
	if mode_label == 'Pure':
		mode_label = 'No Failure'
	all_trials[mode_label] = mode_trials

# datafiles = [x.name for x in Path(directories).expanduser().glob('*.csv')]
# print('Datafiles: {}'.format(datafiles))

all_data = pd.DataFrame()

for mode in all_trials:
	print(mode)
	print(all_trials[mode])

	datafiles = all_trials[mode]

	for i, trial in enumerate(datafiles):	
		# valvegroup = trial.split('/')[-1].split('_')[0]		# Kinda redundant at this point since we have 'mode'
		movement_type = trial.split('/')[-1].split('_')[1]
		trial_number = trial.split('_')[2].split('.')[0]
		# trial_name = movement_type + ' ' + str(trial_number)

		plot_title = 'Comparison of Resultant Motion During {}'.format(movement_type)

		meas, x_filt = estimate_pose(trial, delim_whitespace=False)		# Returns the measured data and Kalman state esimates for the given trial

		# trial_data = pd.concat([meas, x_filt, pd.DataFrame([trial_number for x in meas.index], columns=['Trial'])], axis=1)
		trial_data = pd.concat([meas, x_filt], axis=1)
		if movement_type == 'Translation':
			time_at_cross = trial_data.loc[trial_data['XY Position'] < 0.005]['Time (s)'].iloc[-1]
			delta_tt = 1 - time_at_cross
		elif movement_type == 'Rotation':
			time_at_cross = trial_data.loc[trial_data['r1 Angle'] < 1]['Time (s)'].iloc[-1]
			delta_tt = 0.5 - time_at_cross
		else:
			print('No movement type specified!')
			quit()
		
		
		trial_data['Time (s)'] = trial_data['Time (s)'] + delta_tt
		
		trial_data['Time (s)'] = trial_data['Time (s)'].round(2)	# Round the times to maybe get plotting to work correctly
		trial_data['Trial'] = trial_number
		trial_data['Mode'] = mode

		all_data = all_data.append(trial_data, ignore_index=True)

# Sync data

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


if dev_run:
	fig, axs = plt.subplots(sharex=True, dpi=150, figsize=[6, 4.5])
	fig.suptitle(plot_title, y=0.98, fontsize=12)

	if movement_type == 'Translation':
		sns.lineplot(ax=axs, x="Time (s)", y="XY Position", palette="colorblind", hue='Mode', data=all_data, legend='full')
	elif movement_type == 'Rotation':
		sns.lineplot(ax=axs, x="Time (s)", y="r1 Angle", palette="colorblind", hue='Mode', data=all_data, legend='full')
	else:
		print('No movement type specified!')
		quit()


else:
	fig5, axs = plt.subplots(3, 2, sharex=True, dpi=300, figsize=[6, 4.5])
	fig5.suptitle(plot_title, y=0.98, fontsize=12)
	# fig5.suptitle('Two Consecutive Plenum Discharges (No Failure Modes)', y=0.98, fontsize=12)

	axs[0, 0].set_title('Translational Motion', fontsize=9, color='dimgrey', y=1.04)
	axs[0, 1].set_title('Rotational Motion', fontsize=9, color='dimgrey', y=1.04)

	sns.lineplot(ax=axs[0, 0], x="Time (s)", y="XY Position", palette="colorblind", hue='Mode', data=all_data)
	sns.lineplot(ax=axs[1, 0], x="Time (s)", y="XY Velocity", palette="colorblind", hue='Mode', data=all_data)
	sns.lineplot(ax=axs[2, 0], x="Time (s)", y="XY Acceleration", palette="colorblind", hue='Mode', data=all_data)
	sns.lineplot(ax=axs[0, 1], x="Time (s)", y="r1 Angle", palette="colorblind", hue='Mode', data=all_data)
	sns.lineplot(ax=axs[1, 1], x="Time (s)", y="r1 Velocity", palette="colorblind", hue='Mode', data=all_data)
	sns.lineplot(ax=axs[2, 1], x="Time (s)", y="r1 Acceleration", palette="colorblind", hue='Mode', data=all_data)
	
	axs[0, 0].set_ylabel(r'$d$ (m)', fontsize=8)
	axs[1, 0].set_ylabel(r'$\dot{d}$ (m/s)', fontsize=8)
	axs[2, 0].set_ylabel(r'$\ddot{d}$ (m/$s^2$)', fontsize=8)
	axs[0, 1].set_ylabel(r'$\Theta$ (deg)', fontsize=8)
	axs[1, 1].set_ylabel(r'$\omega$ (deg/s)', fontsize=8)
	axs[2, 1].set_ylabel(r'$\alpha$ (deg/$s^2$)', fontsize=8)
	fig5.align_ylabels()

	axs[0, 0].set_ylim(bottom=-0.015, top=0.415)		# tt
	axs[1, 0].set_ylim(bottom=-0.01, top=0.0725)		# ttdot
	# axs[2, 0].set_ylim(bottom=-0.0425, top=0.0425)		# ttdotdot
	axs[2, 0].set_ylim(bottom=-0.0325, top=0.0525)		# ttdotdot
	axs[0, 1].set_ylim(bottom=-90, top=270)				# r
	axs[1, 1].set_ylim(bottom=-62, top=62)				# rdot
	axs[2, 1].set_ylim(bottom=-30, top=30)				# rdotdot



	class ScalarFormatterForceFormat(mpl.ticker.ScalarFormatter):
		def _set_format(self):  # Override function that finds format to use.
			self.format = "%1.1f"  # Give format here

	for ax in axs.flat:
			ax.legend(loc='upper right', fontsize=6, framealpha=0.9)
			
			yfmt = ScalarFormatterForceFormat()
			yfmt.set_powerlimits((0,0))
			ax.yaxis.set_major_formatter(yfmt)
			ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useMathText=True)
			ax.tick_params(axis='y', labelsize=6, pad=0)
			ax.yaxis.offsetText.set_fontsize(6)

			ax.tick_params(axis='x', labelsize=6, pad=0)
			ax.xaxis.label.set_size(8)
			ax.set(xlabel=r'Time $(sec)$')

	# axs[0, 0].legend()
	axs[1, 0].get_legend().set_visible(False)
	axs[2, 0].get_legend().set_visible(False)
	axs[0, 1].get_legend().set_visible(False)
	axs[1, 1].get_legend().set_visible(False)
	axs[2, 1].get_legend().set_visible(False)


plt.subplots_adjust(left=0.09, right=0.98, top=0.88, bottom=0.1, wspace=0.3, hspace=0.2)
plt.xlim(left=0, right=5)
plt.show()

# if __name__ == "__main__":
# 	try:
# 		directory = sys.argv[1]  # Specify directory containing data files you want to analyze together
# 	except:
# 		pass  # Use defaults

# 	try:
# 		dev_run = bool(sys.argv[2])
# 	except:
# 		pass

# 	print(directory)
# 	print('dev run: {}'.format(dev_run))
# 	batch_kalman(directory, dev_run)