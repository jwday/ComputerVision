Dependencies:
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
