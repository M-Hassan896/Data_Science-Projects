
from ultralytics import YOLO
import cv2
from postprocessing import *
# yolo task=detect mode=predict model=C:\Users\Fazy\PycharmProjects\weapon_detection_ui\training_results\football2\weights\best.pt conf=0.55 source=test_imgs

# Set up model and parameter
model = YOLO(r'C:\Users\Fazy\PycharmProjects\weapon_detection_ui\training_results\football2\weights\best.pt')
class_list = model.model.names
scale_show = 100
# Read Video
video = cv2.VideoCapture(0)
video.set(cv2.CAP_PROP_BUFFERSIZE, 30)
        #
video.set(cv2.CAP_PROP_FRAME_WIDTH, float('inf'))
video.set(cv2.CAP_PROP_FRAME_HEIGHT, float('inf'))
# Run Loop
while True :
    ret, frame = video.read()
    if ret :
        results = model.predict(frame)
        labeled_img = draw_box(frame, results[0], class_list)
        display_img = resize_image(labeled_img, scale_show)
        # Show Image
        cv2.imshow('Frame', display_img)
        # Press Q on keyboard to exit
        if cv2.waitKey(25) & 0xFF == ord('q'): break
# Break the loop if not read
    else: break

# When everything done, release
video.release()
# Closes all the frames
cv2.destroyAllWindows()