# This code will capture and display webcam video, and attempt live detection (tracking) of an Aruco marker.
# A note on syntax...
#	- "detection" refers to the act of detecting an Aruco marker in a single frame
#	- "tracking" refers to the act of successively detecting an Aruco marker across a series of time-ordered frames (AKA a video)
#	- They're fundamentally the same thing. One merely refers to doing it in a single frame, while the other refers to doing it over and over again across multiple frames, presumable in a video.
# If an Aruco marker is found in a frame, it will calculate its relative pose and save it to a Python list
# If no Aruco marker is found in a frame, it will fill the list entry with NA's
# Once capture is ended (by pressing the '0' key), the list will be saved as a .csv file.

# CURRENTLY (as of 5/20/2020), saving the video is not supported. This is a work in progress.
# If you want/need the video, then you will have to run the tracking algorithm as a POST-PROCESS (vs. in-situ, as this script does).
# If that's the case, then use "getPoseAruco-postProcess.py" and specify the location of the video you wish to run the algorithm on.

# USAGE:
# You will need to specify the camera calibration file (defaults to '../images/calib_images/calib.yaml') for whatever camera the source video was captured by.
# If you do not have a calibration file for that camera, use the files in the '../utilities/' directory to create one.

# To run, type the following in the unix command line:
# >>> $ python getPoseAruco-inSitu.py [path to camera calibration file]


# FUTURE WORK:
# Save video!!!!!
# Specify save location


import cv2
import cv2.aruco as aruco
import datetime
import numpy as np

marker_side_length = 0.0655  # Specify size of marker. This is a scaling/unit conversion factor. Without it, all of the measurements would just be determined in units of marker length. At 0.0655, this means a single marker is 0.0655 m per side.
calib_loc = '../images/calib_images/calib.yaml'

def get_pose(calib_loc=calib_loc):
	# Here we instantiate a capture object that we'll pass into the Aruco detection algorithm.
	cap = cv2.VideoCapture(0) 											# '0' means USB device 0
	fps = cap.get(5)													# Get framerate? Not sure how this will turn out for a physical device
	datetime_stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")	# Format datetime stamp to label unique data files

	######################
	# Define the codec and create VideoWriter object
	fourcc = cv2.VideoWriter_fourcc(*'mp4v')
	out = cv2.VideoWriter(datetime_stamp + 'output.mp4', fourcc, fps, (1280,720))
	######################

	# load in camera matrix parameters for pose
	# File storage in OpenCV
	file = open(datetime_stamp + "_datafile.csv", "w")					

	# Note : we also have to specify the type
	# to retrieve otherwise we only get a 'None'
	# FileNode object back instead of a matrix
	cv_file = cv2.FileStorage(calib_loc, cv2.FILE_STORAGE_READ)
	cameraMatrix = cv_file.getNode("camera_matrix").mat()
	distCoeffs = cv_file.getNode("dist_coeff").mat()

	start_time = datetime.datetime.utcnow().timestamp()					# Format start time to be used for data point time stamps

	def draw(img, corners, imgpts):										# I don't think this function is used?
		corner = tuple(corners[0].ravel())
		img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
		img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
		img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
		return img

	output = []

	while(True):
		# Capture frame-by-frame
		ret, frame = cap.read()
		#print(frame.shape) #960x1280 rgb 
		
		frame = cv2.GaussianBlur(frame, (11,11), 0)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250) #6X6 250 very common 
		parameters =  aruco.DetectorParameters_create()
		#print(parameters)
	
		try:
			# lists of ids and the corners belonging to each id
			# this is the function that does all the hard work of actually detecting markers
			corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

			# draw/highlight detected markers
			# this is the function that highlights a detected marker(s)
			# the inputs are the image ('frame'), the detected corners ('corners'), and the ids of the detected markers ('ids')
			# this function is only provided for visualization and its use can be omitted without repercussion
			aruco.drawDetectedMarkers(frame, corners, ids)
			
			# this is the part where we estimate pose of each marker
			# we need to use the camera calibration information in order to correctly estimate the pose after correcting for camera distortion
			# the camera pose with respect to a marker is the 3d transformation FROM the marker coordinate system TO the camera coordinate system
			# it is specified by a rotation and a translation vector (rvec and tvec, respectively)
				# The 'corners' parameter is the vector of marker corners returned by the detectMarkers() function.
				# The second parameter is the size of the marker side in meters or in any other unit. Note that the translation vectors of the estimated poses will be in the same unit
				# cameraMatrix and distCoeffs are the camera calibration parameters that need to be known a priori.
				# rvecs and tvecs are the rotation and translation vectors respectively, for each of the markers in corners.
			rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, marker_side_length, cameraMatrix, distCoeffs)
			print(rvec[0][0], "     ", tvec[0][0])  # only print last element because....not sure why
			
			#  The aruco module provides a function to draw the axis onto the image, so pose estimation can be checked:
			# image is the input/output image where the axis will be drawn (it will normally be the same image where the markers were detected).
			# cameraMatrix and distCoeffs are the camera calibration parameters.
			# rvec and tvec are the pose parameters whose axis want to be drawn.
			# The last parameter is the length of the axis, in the same unit that tvec (usually meters)
			aruco.drawAxis(frame, cameraMatrix, distCoeffs, rvec, tvec, 2*marker_side_length) #Draw Axis

			now = datetime.datetime.utcnow().timestamp() - start_time
			# output = "{0} {1} {2} {3}\n".format(now, tvec[0][0][0], tvec[0][0][1], tvec[0][0][2])
			# file.write(output)
			
			# write all three dimensions
			output.append("{0} {1} {2} {3} {4} {5} {6}\n".format(now, rvec[0][0][0], rvec[0][0][1], rvec[0][0][2],  tvec[0][0][0], tvec[0][0][1], tvec[0][0][2]))
			
			
		except Exception as E:
			#print(E)
			# now = datetime.datetime.utcnow().timestamp() - start_time
			# output.append("{0} {1} {2} {3} {4} {5} {6}\n".format(now, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan))
			# file.write(output)
			
			#write all three dimensions
			# output2 = "{0} {1} {2}\n".format(np.nan, np.nan, np.nan)
			# file2.write(output2)
			pass
			
		############### COME BACK TO THE VIDEO WRITING 
		# out.write(frame)    
		# Display result
		cv2.imshow('Detected Aruco Markers', frame)
		if cv2.waitKey(1) & 0xFF == ord('0'):
			output = ''.join(output)
			file.write(output)
			break
	# When everything done, release the capture
		
	file.close()
	cap.release()
	# out.release()
	cv2.destroyAllWindows()


if __name__ == "__main__":
	try:
		# Specify location of calibration file you wish to use. It will save calibration data in the same location.
		calib_loc = sys.argv[1]
	except:
		pass

	get_pose(calib_loc)
