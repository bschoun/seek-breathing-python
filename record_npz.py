#!/usr/bin/env python3

import time
import numpy as np
from seekcamera import (
    SeekCameraIOType,
    SeekCameraManager,
    SeekCameraManagerEvent,
    SeekCameraFrameFormat,
    SeekCameraShutterMode,
)

width = 320
height = 240
recording_seconds = 30
countdown_seconds = 5
inhale_exhale_duration = 4

# Start with inhaling
exhale = False

def on_frame(camera, camera_frame, _user_data):
    """Async callback fired whenever a new frame is available.

    Parameters
    ----------
    camera: SeekCamera
        Reference to the camera for which the new frame is available.
    camera_frame: SeekCameraFrame
        Reference to the class encapsulating the new frame (potentially
        in multiple formats).
    file: TextIOWrapper
        User defined data passed to the callback. This can be anything
        but in this case it is a reference to the open CSV file to which
        to log data.
    """
    # Get the frame
    frame = camera_frame.thermography_float
    global exhale 
    
    # Write to the user data
    _user_data[0].append(frame.data.copy())
    _user_data[1].append(time.time())
    _user_data[2].append(exhale)


def on_event(camera, event_type, event_status, _user_data):
    """Async callback fired whenever a camera event occurs.

    Parameters
    ----------
    camera: SeekCamera
        Reference to the camera on which an event occurred.
    event_type: SeekCameraManagerEvent
        Enumerated type indicating the type of event that occurred.
    event_status: Optional[SeekCameraError]
        Optional exception type. It will be a non-None derived instance of
        SeekCameraError if the event_type is SeekCameraManagerEvent.ERROR.
    _user_data: None
        User defined data passed to the callback. This can be anything
        but in this case it is None.
    """
    print("{}: {}".format(str(event_type), camera.chipid))

    if event_type == SeekCameraManagerEvent.CONNECT:
        # Start streaming data and provide a custom callback to be called
        # every time a new frame is received.
        
        #Set up manual triggering
        #camera.shutter_mode(SeekCameraShutterMode.AUTO)
        #camera.shutter_trigger()

        # When there's a frame available, run on_frame()
        camera.register_frame_available_callback(on_frame, _user_data)

        # Start the capture session
        camera.capture_session_start(SeekCameraFrameFormat.THERMOGRAPHY_FLOAT)

    elif event_type == SeekCameraManagerEvent.DISCONNECT:
        camera.capture_session_stop()

    elif event_type == SeekCameraManagerEvent.ERROR:
        print("{}: {}".format(str(event_status), camera.chipid))

    elif event_type == SeekCameraManagerEvent.READY_TO_PAIR:
        return


def main():

    # Connect to camera, start collecting frames
    with SeekCameraManager(SeekCameraIOType.USB) as manager:

        # We want both frames and timestamps for accurate playback
        frames = []
        timestamps = []
        exhaling = []

        global exhale

        # Start listening for events.
        manager.register_event_callback(on_event, [frames, timestamps, exhaling])

        # Reset the start time
        start = time.time()
        end = start
        s = 0
        
        print("INHALE")

        while (end - start < recording_seconds):

            # Time elapsed since we started
            _v = end - start

            if (int(_v) > s):
                s = int(_v)
                if s % inhale_exhale_duration == 0:
                    exhale = not exhale
                    if exhale:
                        print("EXHALE")
                    else:
                        print("INHALE")
                elif s % inhale_exhale_duration == (inhale_exhale_duration - 1):
                    print("and...")
                else:
                    print("...")
            end = time.time()

        # Save the frames and timestamps in NPZ format
        print("Saving frames...")
        f = np.array(frames)
        t = np.array(timestamps)
        e = np.array(exhaling)
        np.savez("frames.npz", frames=f, timestamps=t, exhaling=e)
    

if __name__ == "__main__":

    start = time.time()
    end = start
    s = 0

    # This is the countdown before the recording begins. Prints in the terminal
    print("Starting recording in %d..." % countdown_seconds)
    while (end - start < countdown_seconds):

        # Time elapsed since we started
        _v = end - start

        # Comare this to s, which is an integer number of seconds that have passed
        # When we increase to the next integer, print the next number
        if (int(_v) > s):
            s = int(_v)
            print("%d..." % (countdown_seconds-s))
        end = time.time()

    main()