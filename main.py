import argparse
import cv2
import time
import imutils
from imutils.video import VideoStream
from datetime import datetime
import epd7in5_V2
from PIL import Image


PAUSE_BETWEEN_PHOTOS_SECONDS = 30
FACE_DETECTED_DURATION_SECONDS = 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", help="Path to (optional) video file")
    args = vars(ap.parse_args())

    # Use webcam if video not provided
    if not args.get("video", False):
        print("Webcam")
        video_stream = VideoStream(src=0, resolution=(800, 480)).start()
    else:
        print(f"Video: {args['video']}")
        video_stream = cv2.VideoCapture(args["video"])
    time.sleep(2)

    # https://github.com/opencv/opencv/tree/master/data/haarcascades
    face_cascade = cv2.CascadeClassifier(
        "haarcascades/haarcascade_frontalface_default.xml"
    )
    last_photo_datetime = datetime.now()
    face_detected_timer = datetime.now()

    just_finished_pause = True

    # Initialise and clear the e-ink screen
    epd = epd7in5_V2.EPD()
    epd.init()

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
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        # If we've already taken a photo, don't take another one for a given
        # amount of time.
        time_since_last_photo = datetime.now() - last_photo_datetime
        if (
            len(faces)
            and time_since_last_photo.total_seconds() > PAUSE_BETWEEN_PHOTOS_SECONDS
        ):
            # Only take a photo if we've detected a face for more than a given
            # amount of time.
            if just_finished_pause:
                print(
                    f"Wait between photos elapsed ({PAUSE_BETWEEN_PHOTOS_SECONDS}) - ready to detect faces"
                )
                just_finished_pause = False

            face_detected_duration = datetime.now() - face_detected_timer
            if face_detected_duration.total_seconds() > FACE_DETECTED_DURATION_SECONDS:
                # We've detected a face for longer than the threshold - take
                # a photo
                print("Face found - taking photo")
                last_photo_datetime = datetime.now()
                gray_scale = cv2.resize(frame, (800, 480))
                gray_scale = cv2.cvtColor(gray_scale, cv2.COLOR_BGR2GRAY)
                image_pillow = Image.fromarray(gray_scale)
                # Dither the image into a 1 bit bitmap (Just zeros and ones)
                image_pillow = image_pillow.convert(
                    mode="1", dither=Image.FLOYDSTEINBERG
                )

                # Clear the e-ink display and show the image
                # epd.Clear()
                print("Updating display")
                epd.display(epd.getbuffer(image_pillow))
                time.sleep(5)
                # epd.sleep()

        else:
            face_detected_timer = datetime.now()

    # Clean up the video file / webcam
    if not args.get("video", False):
        video_stream.stop()
    else:
        video_stream.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
