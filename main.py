from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time


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

    face_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")
    smile_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_smile.xml")

    while True:
        # Get the current frame
        frame = video_stream.read()
        # Handle the frame from the video file or the webcam
        frame = frame[1] if args.get("video", False) else frame
        # If we reach no frames, we have reached the end of the video
        if frame is None:
            print("Frame is none")
            break

        # Resize, blur and convert to grayscale
        frame = imutils.resize(frame, width=240)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # # Detect smiles in each face
        # smiles = smile_cascade.detectMultiScale(
        #     gray,
        #     scaleFactor=3,
        #     minNeighbors=5,
        #     minSize=(30, 30),
        #     flags=cv2.CASCADE_SCALE_IMAGE
        # )
        # # Draw a rectangle around smiles
        # for (x, y, w, h) in smiles:
        #     cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Show the frame on screen
        cv2.imshow("Frame", frame)
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