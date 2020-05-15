# This script is used to extract and print the pose estimation of an Aruco marker in a single image.
# First, the script will run through the source image, attempt to detect a marker, then draw a coordinate frame on the marker.
# Second, the script will resize the video (specified by the 'out_res' variable) and save a copy of the video in the same directory as the source video.
# Lastly, a .csv file of time, translation, and rotation will be saved in the same directory as the source video.

# Bookoodles of credit and praise go to Adriana Henriquez for figuring out how to do this stuff and writing the initial framework of the code.

# Last update: 4/1/2020

# PRE-USAGE, OPTIONAL:
# Use ffmpeg to convert recorded video to acceptable size, framerate, etc., lest the processing portion of this take 5ever.
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
# Type the following in the unix command line:
# >>> $ ipython -i get_pose_specify_loc [full path to camera calibration file] [full path to source video with aruco marker]
# [full path to camera calibration file] and [full path to source video with aruco marker] MUST be specified

# FUTURE WORK:
# If either paths are not specified, default to local directory
# Allow other options to be specified (such as output video resolution)

import cv2
import cv2.aruco as aruco
import datetime
import numpy as np
import sys
import pandas as pd
import os
import sys

width = int(1920)
height = int(1080)

# Default locations
calib_loc = '/home/josh/ComputerVision/calib_images/device_nokia7/calib.yaml'
image_loc = '/home/josh/ComputerVision/images/test_frame.png'

def get_pose(calib_loc=calib_loc, image_loc=image_loc):
	if os.path.isfile(image_loc) and os.path.isfile(calib_loc):
		print('')
		print('Running Aruco detection on a single image with specified location and calibration:')
		print('	Image location: {0}'.format(image_loc))
		print('	Calibration file location: {0}'.format(calib_loc))
		print('')
	else:
		print('')
		print('*** One or more of your file paths are invalid. ***')
		print('')
		sys.exit()

	# Import calibration items (camera matrix and distortion coefficients)
	print('Importing calibration file...')
	calib_file = cv2.FileStorage(calib_loc, cv2.FILE_STORAGE_READ)	# Load in camera matrix of distortion correction parameters for the camera used to capture source video, for pose estimation
	cameraMatrix = calib_file.getNode("camera_matrix").mat()		# Grab the camera calibration matrix
	distCoeffs = calib_file.getNode("dist_coeff").mat()				# Grab the camera distortion matrix

	# Read the specified image and open a window to display it
	print('Drawing image...')
	cap = cv2.imread(image_loc)
	cv2.namedWindow('source', cv2.WINDOW_NORMAL)
	cv2.startWindowThread()
	cv2.resizeWindow('source', width, height)
	cv2.imshow('source', cap)
	while not (cv2.waitKey(1) & 0xFF == ord('q')):
		pass
	cv2.destroyAllWindows()

	# Search the image for Aruco marker(s), draw a border around it, then open another window to display it
	print('Drawing Aruco marker...')
	blur = cv2.GaussianBlur(cap, (11, 11), 0)
	gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
	aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
	parameters = aruco.DetectorParameters_create()
	corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
	cap_marked = aruco.drawDetectedMarkers(cap, corners, ids)
	cv2.namedWindow('marker', cv2.WINDOW_NORMAL)
	cv2.resizeWindow('marker', width, height)
	cv2.imshow('marker', cap_marked)
	while not (cv2.waitKey(1) & 0xFF == ord('q')):
		pass
	cv2.destroyAllWindows()

	# Estimate the marker pose, draw a coordinate frame on it, then open yet another window to display it
	print('Drawing pose axes...')
	rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, 0.0655, cameraMatrix, distCoeffs)
	print('')
	print('tvecs: {0}'.format(tvec[0][0]))
	print('rvecs: {0}'.format(rvec[0][0]))
	print('')
	cap_drawn = aruco.drawAxis(cap, cameraMatrix, distCoeffs, rvec, tvec, 2*0.0655)
	cv2.namedWindow('axes', cv2.WINDOW_NORMAL)
	cv2.resizeWindow('axes', width, height)
	cv2.imshow('axes', cap_drawn)
	while not (cv2.waitKey(1) & 0xFF == ord('q')):
		pass
	cv2.destroyAllWindows()

	# pose_transformation = []
	# pose_transformation.append([frame_time, rvec[0][0][0], rvec[0][0][1], rvec[0][0][2],  tvec[0][0][0], tvec[0][0][1], tvec[0][0][2]])

	# Display the result with reduced size (in case your source video has larger resolution than your monitor)
	# Resize the frame from the source video and save it as a new object 'b'
	# b = cv2.resize(frame, out_res, fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
	# Create a window to display the modified frames in
	# cv2.namedWindow('Detected Aruco Markers', cv2.WINDOW_AUTOSIZE)
	# Resize the window by explicitly defining its resolution (without this it MAY appear teeny-tiny for no apparent reason)
	# cv2.resizeWindow('Detected Aruco Markers', out_res)
	# cv2.imshow('Detected Aruco Markers', b)										# SHOW ME WHAT YOU GOT.

	# ...and write the result to the output video object 'out'
	# out.write(b)

	# Press 'q' to quit early. Don't worry, the video has already been written to 'out'.
	# if cv2.waitKey(1) & 0xFF == ord('q'):
		# break

		# When everything done, release the capture
		# headers = ['Time (s)', 'r1', 'r2', 'r3', 't1', 't2', 't3']
		# df = pd.DataFrame(pose_transformation, columns=headers)
		# df.to_csv('/'.join(video_loc.split("/")[:-1]) + "/datafile.csv", index=False)
		# file.close()
		# cap.release()
		# out.release()
		# cv2.destroyAllWindows()
		# break

		# print('Process complete. Video saved to {0}'.format(out_loc))


if __name__ == "__main__":		
	try:
		# Specify location of calibration file you wish to use. It will save calibration data in the same location.
		calib_loc = sys.argv[1]
		# Specify location of video file containing the aruco marker which you wish to extract the pose of.
		image_loc = sys.argv[2]
	except:
		pass

	get_pose(calib_loc, image_loc)

		