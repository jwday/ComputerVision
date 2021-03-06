Credit where credit is due:
----------------------------
The computer vision portion of this code was inspired by and expanded upon from njanirudh's Aruco Tracker repo: https://github.com/njanirudh/Aruco_Tracker


Dependencies:
-------------------
 - For capturing with a camera device, you will need a native Linux OS such as Ubuntu or Raspbian (WSL does not work with USB cameras *yet*)
 - Python 3.x
 - opencv-contrib-python (user:~$ pip3 install opencv-contrib-python)
    - This will also install numpy, which is required
    - This distro includes Aruco
 
 Test out your installation with the following steps:
 
 1) Open an interactive python shell
 
```
user@pi:~$ python3
```
```
Python 3.7.4 (default, Aug 13 2019, 20:35:49) 
[GCC 7.3.0] :: Anaconda, Inc. on linux
Type "help", "copyright", "credits" or "license" for more information.
>>>
```

2) Import cv2

```
>>> import cv2
```

3) Import cv2.aruco

```
>>> import cv2.aruco
```

If you run into any problems, follow the steps shown here:
https://blog.piwheels.org/how-to-work-out-the-missing-dependencies-for-a-python-package/


Instructions
--------------
**Computer Vision:**
1) Connect camera 
2) Run utilities/captureSingleFrame-webcam-v1.01.py
	- WILL NOT WORK IN WSL DUE TO LACK OF USB FUNCTIONALITY (for now)
    - Print out checkerboard.png (located in ../images/) on a standard piece of paper and affix it to a rigid surface (such as a clipboard).
    - Press 's' to take a picture.
    - Press 'q' to quit.
    - Take at least 10 pictures of the checkerboard at different distances and angles from the camera (more is better).
3) Run utilities/calibration_checkerboard.py
    - If the code does not work, run captureSingleFrame-webcam-v1.01.py again and take more images. You may need as much as 100 images!
    - This code will create a save a file called test.yaml inside ```../images/calib_images``` (or whatever location you specify). This file contains all of the camera distortion matrices needed to correct for intrinsic and extrinsic distortion.
4) Run one of the Aruco detection scripts in aruco/
    - If you are using a new aruco marker, measure the width of the marker in meters with a caliper, and change the marker_size value. 
    - Time, x, y, and z positions will be written into datafile.csv, and the rotation vectors for roll, pitch, and yaw, will be written in the rvec.csv file.
    - The video will be written as an mp4 in output.mov. 
    - Press '0' to end the program. 

**Post-Processing:**
1) Run velocity.py 
    - This plots the position and velocity of the data from datafile.csv. This data is noisy.
2) Run Kalman.py
    - Implements a 1-D constant velocity Kalman filter to remove sensor noise.


Demonstration
--------------
The animated .gif below illustrates the result of the Aruco fiducial marker detection process. The original video was captured at a resolution of 720 x 1280 at 30 frames/second. The Python script `aruco/getPoseAruco-postProcess.py` was used to process the original video. The process exports a copy of the video with the Aruco marker highlighted (as shown) and a .csv dataset (also shown) of the 3D position and 3D orientation of the Aruco marker relative to the center of the camera.

![Sample rotation](/A1+A2_Rotation_4_aruco.gif)

![Sample dataset](/A1+A2_Rotation_dataset.png)

The timeseries dataset of position and orientation is fed into the Python-based Kalman filtering script `kalman/batch_kalman_3D.py`, which batch-processes any .csv file found in the specified directory. The Kalman filter combines a simple kinematic model with the real data from the Aruco measurements to produce a statistically likely estimation of position and orientation vs. time of the air bearing platform. The results of multiple trials for the same expected motion are combined into a single image (as shown):

![Sample results](/A1+A2_Rotation.png)

For this particular example, an attempted rotation with valves A1 and A2 failed open results in a large coupled translation + rotation. Without the valve failures included, this maneuver should have only produced pure rotational motion with no translational component.
