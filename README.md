Credit where credit is due:
----------------------------
Most of this code was taken from njanirudh's Aruco Tracker repo: https://github.com/njanirudh/Aruco_Tracker


Dependencies:
-------------------
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
2) Run webcam-capture-v1.01.py
    - Print out checkerboard.png (located in ./images/) on a standard piece of paper and affix it to a rigid surface (such as a clipboard).
    - Take at least 10 pictures of the checkerboard at different distances and angles from the camera.
    - Press 's' to take a picture.
    - Press 'q' to quit.
3) Run calibration_checkerboard.py
    - If the code does not work, run take_picture.py again and take more images. You may need up to 100 images.
    - This code will create a save a file called calib.yaml. This file contains all of the camera distortion matrices needed to correct for intrinsic and extrinsic distortion.
4) Run  aruco_reader.py
    - If you are using a new aruco marker, measure the width of the marker in meters with a caliper, and change the marker_size value. 
    - Time, x, y, and z positions will be written into datafile.csv, and the rotation vectors for roll, pitch, and yaw, will be written in the rvec.csv file.
    - The video will be written as an mp4 in output.mov. 
    - Press '0' to end the program. 

**Post-Processing:**
1) Run velocity.py 
    - This plots the position and velocity of the data from datafile.csv. This data is noisy.
2) Run Kalman.py
    - Implements a 1-D constant velocity Kalman filter to remove sensor noise. 
