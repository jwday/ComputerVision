# Capture a single image from the webcam when a button is pressed then save the image
# Useful for capturing a relatively small number of still frames (such as for capturing calibration images)

import cv2
from datetime import datetime

key = cv2. waitKey(1)
webcam = cv2.VideoCapture(0)
img_int = 0

while True:
    check, frame = webcam.read()
    cv2.imshow("Capturing", frame)
    key = cv2.waitKey(1)
    
    if key == ord('s'):
        print("Capturing image...")
        img_filename = 'capture_' + str(img_int) + '.jpg'
        now = datetime.now()
        dt_string = now.strftime("%H:%M:%S")
        cv2.imwrite(filename='calib_images' + img_filename, img=frame)
        img_int += 1
        print("Processing image...")
        img_ = cv2.imread(img_filename, cv2.IMREAD_ANYCOLOR)
        print("Converting RGB image to grayscale...")
        gray = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
        print("Converted RGB image to grayscale...")
        print("Image {} saved at {}.".format(img_filename, dt_string))
        
    elif key == ord('q'):
        print("Turning off camera.")
        webcam.release()
        print("Camera off.")
        print("Program ended.")
        cv2.destroyAllWindows()
        break