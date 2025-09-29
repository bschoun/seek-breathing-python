import cv2
import os.path
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Thermal breathing analysis")
    parser.add_argument('filename')
    return parser.parse_args()

''' Gets and thresholds mask for the images. '''
def get_mask(filename):
    mask = cv2.imread(filename, 0)
    _ret, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return mask


''' Ensures filename ends with the right extenion and exists.'''
def check_filename(filename):
    if not filename.endswith(".npz"):
        print("Invalid file extension, exiting.")
        return False
    if not os.path.isfile(filename):
        print("This file does not exist")
        return False
    return True

def get_all_npz(directory_path): 
    try:
        # Get all entries (files and directories) in the specified path
        entries = os.listdir(directory_path)
        
        # Filter out only the files
        files = [entry for entry in entries if os.path.isfile(os.path.join(directory_path, entry))]
        files = [entry for entry in files if entry.endswith('.npz')]
        return files
    except FileNotFoundError:
        print(f"Error: Directory not found at '{directory_path}'")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

''' Finds indices in a list that meet a specific condition. '''
def find_indices(list_obj, condition):
    return [index for index, element in enumerate(list_obj) if condition(element)]

''' Loads data '''
def load_data(filename):
    print("loading data")
    # Load the data
    data = np.load(filename)
    frames = data["frames"]
    timestamps = data["timestamps"]
    timestamps -= timestamps[0]     # Subtract the intital time stamp from all timestamps, converting to relative time

    exhaling_data = None
    if "exhaling" in data:
        exhaling_data = data["exhaling"]
        
    # Make sure the timestamps and frames are the same length
    if len(timestamps) > len(frames):
        timestamps = timestamps[:len(frames)]

    # Trim exhaling if needed
    if exhaling_data is not None:
        if len(exhaling_data) > len(frames):
            exhaling_data = exhaling_data[:len(frames)]

    return frames, timestamps, exhaling_data