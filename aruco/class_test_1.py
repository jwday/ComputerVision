# This script is my first attempt at creating class-based functionality to implement MQTT-based ArUco recording
# The objective is to have this script be executed just after the server is started.
# Upon execution, the following sequence of events is desired:
# 	1. The script, as an MQTT client, should connect to the MQTT server
# 	2. The script, as an MQTT client, will sit idle until the proper command (topic + message) is received
#	3. Once the message is received, the script will create a new recording (as an object) and begin recording data from the ArUco-based CV input
#	4. Simultaneously, the recording object will record all propulsive inputs from the server (possibly in the form of 'FWD', 'BCK', 'LT', 'RT', 'CW', 'CCW')
#	5. The recording will stop on command when the proper message + topic is received
#	6. The recording will be saved and the object terminated (is that a thing?)

from datetime import datetime
import cv2
import cv2.aruco as aruco
import os
import sys
import paho.mqtt.client as mqtt

# -----------------------------------------------------------------------------
# MQTT SETUP
# -----------------------------------------------------------------------------
# The callback for when the client receives a CONNACK response from the server.
def on_connect(client, userdata, flags, rc):
	print("Connected with result code "+str(rc))
	client.subscribe([("propel",2), ("timedPropel",2), ("CV",2)])


# The callback for when a PUBLISH message is received from the server.
def on_message(client, userdata, msg):
	global recordBool
	msg.payload = msg.payload.decode("utf-8")
	print("incoming: " + msg.topic + " " + str(msg.payload))

	if msg.topic == "singleValveOn":
		valve_status = str(msg.payload)
	
	elif msg.topic == "singleValveOff":
		valve_status = 'Off'
	
	if msg.topic == "CV":
		if msg.payload == "recordON":
			print("Go!")
			recordBool = True
			track_and_record(calib_loc, video_loc)
		if msg.payload == "recordOFF":
			print("Stop!")
			recordBool = False

# -----------------------------------------------------------------------------
# Camera setup
# -----------------------------------------------------------------------------
# Set up the camera and recording parameters. This should NOT define the filename for the recording -- that will be done when the Recording object is generated.
def setup_recording(calib_loc=calib_loc, video_loc=video_loc):
	if os.path.isfile(calib_loc) & os.path.isdir(video_loc):
		print('')
		print('Readying Aruco detection on a live webcam feed using the following settings:')
		print('	Calibration file location: {0}'.format(os.path.abspath(calib_loc)))
		print('	Video save location: {0}'.format(os.path.abspath(video_loc)))
		print('')
	else:
		print('')
		print('*** One or more file path is invalid. ***')
		print('	Specified calibration file location: {0}'.format(os.path.abspath(calib_loc)))
		print('')
		sys.exit()

	# Import calibration items (camera matrix and distortion coefficients)
	print('Importing calibration file...')	
	calib_file = cv2.FileStorage(calib_loc, cv2.FILE_STORAGE_READ)	# Load in camera matrix of distortion correction parameters for the camera used to capture source video, for pose estimation
	cameraMatrix = calib_file.getNode("camera_matrix").mat()		# Grab the camera calibration matrix
	distCoeffs = calib_file.getNode("dist_coeff").mat()				# Grab the camera distortion matrix
	print('Done!')
	print('')



# -----------------------------------------------------------------------------
# Recording Object
# -----------------------------------------------------------------------------
class Recording:
	def __init__(calib_loc=calib_loc, video_loc=video_loc):
		datetime_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")	# Format datetime stamp to label unique data files

		# Here we instantiate a capture object that we'll pass into the Aruco detection algorithm. Also we grab the total number of frames to provide the user with a % completion measure during processing.
		cap = cv2.VideoCapture(0)										# Instantiate video capture object 'cap' using DEVICE 0
		fps = cap.get(cv2.CAP_PROP_FPS)									# Grab the FPS of the source video so you can properly calculate elapsed process time
		source_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) 			# Grab the source video width
		source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))			# Grab the source video height
		print('Source resolution: {0} x {1} px'.format(source_width, source_height))
		print('Source fps: {}'.format(fps))
		print('')

		# Here we define the codec and create a VideoWriter object to resave the video (with a coordinate frame drawn on the Aruco marker and reduced in size).
		fourcc = cv2.VideoWriter_fourcc(*"XVID")						# Specify the codec used to write the new video
		out_loc = video_loc + datetime_stamp + '_record.avi'			# Save the video in the same directory as the source video, call it 'aruco_output.mp4'
		out_res = (source_width, source_height)							# Output video resolution (NOTE: This is NOT the video that the Aruco marker will be tracked from. The marker will still be tracked from the source video--this is the output that the coordinate axes are drawn on.)
		out = cv2.VideoWriter(out_loc, fourcc, 5, out_res)				# Instantiate an object of the output video (to have a coordinate frame drawn on the Aruco marker and resized)

		print('')
		print('Ready to record and process live video. Processed video will be saved saved to {0}'.format(os.path.abspath(calib_loc)))
		print("Send message 'recordON' to topic 'CV' to begin.")
		print("Send message 'recordOFF' to topic 'CV' to stop.")
		print('')

	def beginRecording(capture, calib_loc=calib_loc, video_loc=video_loc):




if __name__ == "__main__":
	try:
		# Specify location of calibration file you wish to use. It will save calibration data in the same location.
		calib_loc = sys.argv[1]
		# Specify location of video file containing the aruco marker which you wish to extract the pose of.
		video_loc = sys.argv[2]
	except:
		pass

	# Set up MQTT client
	client = mqtt.Client("computervision", protocol=mqtt.MQTTv31)
	client.on_connect = on_connect
	client.on_message = on_message

	# Connect!
	client.connect("localhost", 1883, 60)	# (host, port, keepalive)
	setup_recording(calib_loc, video_loc)

	while True:
		client.loop()