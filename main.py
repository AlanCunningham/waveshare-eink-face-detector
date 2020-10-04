from collections import deque
from imutils.video import VideoStream
from datetime import datetime
import numpy as np
import argparse
import cv2
import imutils
import time


PAUSE_BETWEEN_PHOTOS_SECONDS = 15
FACE_DETECTED_DURATION_SECONDS = 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", help="Path to (optional) video file")
    args = vars(ap.parse_args())

    # Use webcam if video not provided
    if not args.get("video", False):
        print("Webcam")
        video_stream = VideoStream(src=0).start()
    else:
        print(f"Video: {args['video']}")
        video_stream = cv2.VideoCapture(args["video"])
    time.sleep(2)

    # https://github.com/opencv/opencv/tree/master/data/haarcascades
    face_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")
    smile_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_smile.xml")
    last_photo_datetime = datetime.now()
    face_appeared_datetime = datetime.now()

    while True:
        # Get the current frame
        frame = video_stream.read()
        # Handle the frame from the video file or the webcam
        frame = frame[1] if args.get("video", False) else frame
        # If we reach no frames, we have reached the end of the video
        if frame is None:
            print("Frame is none")
            break

        # Resize and convert to grayscale
        resized_image = imutils.resize(frame, width=240)
        gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            # minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        # Draw a rectangle around the faces
        # for (x, y, w, h) in faces:
        #     cv2.rectangle(gray, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # # Detect smiles in each face
        # for face in faces:
        #     # region_of_interest = gray[y_face:y_face + h_face, x_face:x_face + w_face]
        #     smiles = smile_cascade.detectMultiScale(
        #         gray,
        #         scaleFactor=1.7,
        #         minNeighbors=20,
        #         # minSize=(30, 30),
        #         # flags=cv2.CASCADE_SCALE_IMAGE
        #     )
        #     # Draw a rectangle around smiles
        #     for (x_smile, y_smile, w_smile, h_smile) in smiles:
        #         cv2.rectangle(gray, (x_smile, y_smile), (x_smile + w_smile, y_smile + h_smile), (0, 255, 0), 2)

        # If we've already taken a photo, don't take another one for a given
        # amount of time.
        time_since_last_photo = datetime.now() - last_photo_datetime
        if len(faces) and time_since_last_photo.total_seconds() > PAUSE_BETWEEN_PHOTOS_SECONDS:
            # Only take a photo if we've detected a face for more than a given
            # amount of time.
            face_detected_duration = datetime.now() - face_appeared_datetime
            if face_detected_duration.total_seconds() > FACE_DETECTED_DURATION_SECONDS:
                gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                last_photo_datetime = datetime.now()
                cv2.namedWindow("CheerInk", cv2.WND_PROP_FULLSCREEN)
                cv2.setWindowProperty("CheerInk",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
                cv2.imshow("CheerInk", gray_scale)
        else:
            face_appeared_datetime = datetime.now()

        # Show the frame on screen
        # cv2.imshow("Frame", gray)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    # Clean up the video file / webcam
    if not args.get("video", False):
        video_stream.stop()
    else:
        video_stream.release()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()