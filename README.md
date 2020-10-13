# Waveshare E-Ink Face Detector

A Raspberry Pi powered e-ink display that detects faces and displays them as a photo on [Waveshare's 7.5 inch e-ink display](https://www.waveshare.com/7.5inch-e-paper-hat.htm).

# Hardware requirements
- Raspberry Pi
- [Waveshare's 7.5 inch e-ink display](https://www.waveshare.com/7.5inch-e-paper-hat.htm)
- A camera (e.g. the Raspberry Pi camera, USB camera, or a networked video feed)
- A photo frame

# Installation
```
# Clone the repository
git clone git@github.com:AlanCunningham/krpc-scripts.git

# Create a python virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Install the python dependencies using the requirements.txt file provided
pip install -r requirements.txt
```

# Running script
If you're using the Raspberry Pi camera or a USB camera:
```
(venv)$ python main.py
```
If you're using a networked camera:
```
(venv)$ python main.py --video url_to_video_feed
```

The script will look for faces and take a photo if one is detected for more than
2 seconds and display it on the screen.  It'll then go to sleep for a given amount
of time to prevent it updating while the face is still there.
