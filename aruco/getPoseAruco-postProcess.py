# This script is used to extract and save pose estimation of an Aruco marker in a source video and also save a copy of the video with the Aruco marker pose axes drawn.
# First, the script will run through the source video, attempt to detect a marker, then draw a coordinate frame on the marker.
# Second, the script will resize the video (specified by the 'out_res' variable) and save a copy of the video in the same directory as the source video.
# Lastly, a .csv file of time, translation, and rotation will be saved in the same directory as the source video.

# Bookoodles of credit and praise go to Adriana Henriquez for figuring out how to do this stuff and writing the initial framework of the code.

# Last update: 5/20/2020

# PRE-USAGE, OPTIONAL:
# Use ffmpeg to convert recorded video to acceptable size, framerate, etc., or else the processing portion of this will take 5ever.
# ex. > $ ffmpeg -ss 5 -i VID_20200328_104239.mp4 -t 84 -an -filter:v "scale=720:-1, transpose=2, fps=10" output.mp4
# ..... will trim video by starting at 5 sec (using -ss parameter) and ending 84 seconds after (using -t parameter)
# ..... will take *.mp4 file type input (using -i parameter)
# ..... will remove audio (using -an parameter)
# ..... will apply multiple filters to the video (using -filter:v paramter followed by a string)
# ........... scale the video to 720 WIDTH (using the 'scale=720:-1' option)
# ........... rotate the video 90 degrees counter-clockwise (using the 'transpose=2' option)
# ........... downsample the framerate to 10 fps (using the 'fps=10' option)
# ..... will save it to an output file (must complete the command with a destination)

# USAGE:
# You will need to specify the camera calibration file (defaults to '../images/calib_images/calib.yaml') for whatever camera the source video was captured by.
# If you do not have a calibration file for that camera, use the files in the '../utilities/' directory to create one.
# You will also need to specifiy the location of the video from which you wish to extract Aruco pose.

# To run, type the following in the unix command line:
# >>> $ python getPoseAruco-postProcess.py [path to camera calibration file] [path to source video with aruco marker]


# FUTURE WORK:
# Allow other options to be specified (such as output video resolution, currently defaults to 1280x720)
# Better filename saves (i.e. include datetime). Currently will just save as "aruco_output.mp4", potentially overwriting any file that currently exists

import cv2
import cv2.aruco as aruco
import datetime
import numpy as np
import sys
import pandas as pd
import os
import sys
import math

marker_side_length = 0.0655  # Specify size of marker. This is a scaling/unit conversion factor. Without it, all of the measurements would just be determined in units of marker length. At 0.0655, this means a single marker is 0.0655 m per side.

# Output video size
height = int(720)
width = int(height * 16/9)

# Default locations
calib_loc = '../images/calib_images/calib.yaml'
video_loc = '../videos/output_720p_10fps.mp4'

def get_pose(calib_loc=calib_loc, video_loc=video_loc):
	if os.path.isfile(video_loc) and os.path.isfile(calib_loc):
		print('')
		print('Running Aruco detection on a video with specified location and calibration:')
		print('	Video location: {0}'.format(video_loc))
		print('	Calibration file location: {0}'.format(calib_loc))
		print('')
	else:
		print('')
		print('*** One or more of your file paths are invalid. ***')
		print('	Specified video location: {0}'.format(video_loc))
		print('	Specified calibration file location: {0}'.format(calib_loc))
		print('')
		if not os.path.isfile(video_loc):
			print('Specified video location is not valid')
		if not os.path.isfile(calib_loc):
			print('Specified calibration file location is not valid')
		print('')
		sys.exit()

	# Here we set everything up by pulling the video directory so we can save a new file to it later. Also we pull the calibration matrices for the camera used to capture the source video. If you use a new camera, you'll have to get a new calibration matrix.
	video_loc_dir = '/'.join(video_loc.split('/')[:-1])+'/'  	# Grab the directory of the source video. (ex. if 'video_loc' is '/mnt/c/Users/Josh/Desktop/Photos/output.mp4', this will return '/mnt/c/Users/Josh/Desktop/Photos/')

	# Import calibration items (camera matrix and distortion coefficients)
	print('Importing calibration file...')	
	calib_file = cv2.FileStorage(calib_loc, cv2.FILE_STORAGE_READ)	# Load in camera matrix of distortion correction parameters for the camera used to capture source video, for pose estimation
	cameraMatrix = calib_file.getNode("camera_matrix").mat()		# Grab the camera calibration matrix
	distCoeffs = calib_file.getNode("dist_coeff").mat()				# Grab the camera distortion matrix
	print('Done!')
	print('')

	# Here we instantiate a capture object that we'll pass into the Aruco detection algorithm. Also we grab the total number of frames to provide the user with a % completion measure during processing.
	cap = cv2.VideoCapture(video_loc)								# Instantiate video capture object 'cap'
	fps = cap.get(cv2.CAP_PROP_FPS)									# Grab the FPS of the source video so you can properly calculate elapsed process time
	no_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))				# Grab the total number of frames in the source video
	source_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) 			# Grab the source video width
	source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))			# Grab the source video height
	print('Source size: {0} x {1} px'.format(source_width, source_height))
	print('Source fps: {}'.format(fps))
	print('Source frames: {}'.format(no_frames))
	print('')

	# Here we define the codec and create a VideoWriter object to resave the video (with a coordinate frame drawn on the Aruco marker and reduced in size).
	fourcc = cv2.VideoWriter_fourcc(*"mp4v")						# Specify the codec used to write the new video
	out_loc = video_loc_dir + 'aruco_output.mp4'					# Save the video in the same directory as the source video, call it 'aruco_output.mp4'
	out_res = (width, height)										# Output video resolution (NOTE: This is NOT the video that the Aruco marker will be tracked from. The marker will still be tracked from the source video--this is the output that the coordinate axes are drawn on.)
	out = cv2.VideoWriter(out_loc, fourcc , fps, out_res)			# Instantiate an object of the output video (to have a coordinate frame drawn on the Aruco marker and resized)

	# Set up the data storage variables to be appended or updated through the algorithm's loop.
	pose_transformation = []										# This is the list which will store the pose transformation values (3x translation, 3x rotation)
	frame_count = 0													# Frame counter, to be used to provide the user with a % completion measure during processing

	print('')
	print('Processing video. Press \'q\' to quit. Processed video will be saved saved to {0}'.format(out_loc))
	print('')

	while(True):
		# Capture frame-by-frame
		ret, frame = cap.read()									# Read the next frame in the buffer and return it as an object
		frame_count += 1										# Increment the frame counter
		percent_complete = int(100*(frame_count/no_frames))		# Calculate % completion
		print('\rFrame: {0}/{1} ({2}%)'.format(frame_count, no_frames, percent_complete), end='')	# Print % completion, using \r and end='' to overwrite the previously displayed text (so it doesn't spam the terminal)

		if (not ret):											# If cap.read() doesn't return anything (i.e. if you've reached the end of the source video)
			break												# Kill the loop

		blur = cv2.GaussianBlur(frame, (7,7), 0)				# As part of the Aruco marker detection algorithm, we blur the frame
		gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)			# Next, we make the frame grayscale
		aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250) 	# Define the shape of the Aruco marker we are trying to detect (6X6_250 is very common)
		parameters = aruco.DetectorParameters_create()			# Not sure what this step does but Adriana put it in and I trust her
		parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
		# parameters.polygonalApproxAccuracyRate = 0.05
		# parameters.cornerRefinementWinSize = 5
		# parameters.cornerRefinementMinAccuracy = 0.01
		# parameters.perspectiveRemovePixelPerCell = 8
		# parameters.maxErroneousBitsInBorderRate = 0.04
		# parameters.errorCorrectionRate = 0.2
		# parameters.adaptiveThreshWinSizeMin= 170
		# parameters.adaptiveThreshWinSizeMax= 255
		# parameters.adaptiveThreshWinSizeStep= 3
		
		try:
			# Here is the function that does all the hard work of actually detecting markers
			corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)	# List of ids and the corners belonging to each id

			# This is the function that highlights a detected marker(s)
			# The inputs are the image ('frame'), the detected corners ('corners'), and the ids of the detected markers ('ids')
			# This function is only provided for visualization and its use can be omitted without repercussion
			aruco.drawDetectedMarkers(frame, corners, ids)
			
			# This is the part where we actually estimate pose of each marker
			# We need to use the camera calibration information in order to correctly estimate the pose after correcting for camera distortion
			# The camera pose with respect to a marker is the 3d transformation FROM the marker coordinate system TO the camera coordinate system
			# It is specified by a rotation and a translation vector (rvec and tvec, respectively) and stored in RODRIGUES FORM (???)
				# The 'corners' parameter is the vector of marker corners returned by the detectMarkers() function.
				# The second parameter is the size of the marker side in meters or in any other unit. Note that the translation vectors of the estimated poses will be in the same unit
				# cameraMatrix and distCoeffs are the camera calibration parameters that need to be known prior to executing this function.
				# rvecs and tvecs are the rotation and translation vectors respectively.
				# The marker coordinate axes are centered on the middle of the marker, with the Z axis perpendicular to the marker plane.
			rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, marker_side_length, cameraMatrix, distCoeffs)
			dst, jacobian = cv2.Rodrigues(rvec)		# I think this is the rotation matrix...that...rotates from the marker to the camera?
			rot_alpha = math.atan(dst[1][0] / dst[0][0])
			rot_beta = math.atan(dst[2][0] / math.sqrt(dst[2][1]**2 + dst[2][2]**2))
			rot_gamma = math.atan(dst[2][1] / dst[2][2])
			# print('\rRotation Angles: {0}, {1}, {2}'.format(rot_alpha*180/3.141, rot_beta*180/3.141, rot_gamma*180/3.141), end='')	# Print % completion, using \r and end='' to overwrite the previously displayed text (so it doesn't spam the terminal)

			# The aruco module provides a function to draw the coordinate axes onto the image, so pose estimation can be visually verified:
			# Image is the input/output image where the axis will be drawn (it will normally be the same image where the markers were detected).
			# cameraMatrix and distCoeffs are the camera calibration parameters.
			# rvec and tvec are the pose parameters whose axis want to be drawn.
			# The last parameter is the length of the axis, in the same unit that tvec (usually meters)
			aruco.drawAxis(frame, cameraMatrix, distCoeffs, rvec, tvec, 2*marker_side_length) #Draw Axis

			frame_time = frame_count/fps						# Calculate the elapsed time based on which frame (from the source video) you're on and what the FPS of that video is
			
			# Append tvecs and rvecs to the pose_transformation list, to be saved to a csv after the loop is complete
			# pose_transformation.append([frame_time, rvec[0][0][0], rvec[0][0][1], rvec[0][0][2],  tvec[0][0][0], tvec[0][0][1], tvec[0][0][2]])
			pose_transformation.append([frame_time, rot_alpha, rot_beta, rot_gamma,  tvec[0][0][0], tvec[0][0][1], tvec[0][0][2]])
		
			
		except:
			# If any of the functions in the loop throw an error, just write NaNs to this data point and move on
			frame_time = frame_count/fps						# Calculate the elapsed time based on which frame (from the source video) you're on and what the FPS of that video is
			# Append tvecs and rvecs to the pose_transformation list, to be saved to a csv after the loop is complete
			pose_transformation.append([frame_time, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
			pass
			
		# Display the result with reduced size (in case your source video has larger resolution than your monitor)
		b =  cv2.resize(frame, out_res, fx=0, fy=0, interpolation=cv2.INTER_CUBIC)	# Resize the frame from the source video and save it as a new object 'b'
		cv2.namedWindow('Detected Aruco Markers', cv2.WINDOW_AUTOSIZE)				# Create a window to display the modified frames in
		cv2.resizeWindow('Detected Aruco Markers', out_res)							# Resize the window by explicitly defining its resolution (without this it MAY appear teeny-tiny for no apparent reason)
		cv2.imshow('Detected Aruco Markers', b)										# SHOW ME WHAT YOU GOT.

		# ...and write the result to the output video object 'out'
		out.write(b)

		# Press 'q' to quit early. Don't worry, the video has already been written to 'out'.
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	# When everything done, release the capture
	headers = ['Time (s)', 'r1', 'r2', 'r3', 't1', 't2', 't3']
	df = pd.DataFrame(pose_transformation, columns=headers)
	df.to_csv('/'.join(video_loc.split("/")[:-1]) + '/datafile_{0}p_{1}fps.csv'.format(source_height, fps), index=False)
	# file.close()
	cap.release()
	out.release()
	cv2.destroyAllWindows()

	print('')
	print('')
	print('Process complete. Video saved to {0}'.format(out_loc))


if __name__ == "__main__":
	try:
		# Specify location of calibration file you wish to use. It will save calibration data in the same location.
		calib_loc = sys.argv[1]
		# Specify location of video file containing the aruco marker which you wish to extract the pose of.
		video_loc = sys.argv[2]
	except:
		pass

	get_pose(calib_loc, video_loc)