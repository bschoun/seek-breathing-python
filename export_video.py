import numpy as np
import cv2
import sys
import os.path
import matplotlib.pyplot as plt
import time
import argparse


folder_name = "data/data_final/"

classes = ["gale", "waft", "other", "none"]

gale_count = 0
waft_count = 0
other_count = 0
none_count = 0

confidence_level = 0.25

def get_files_in_directory(directory_path):
    """
    Lists all files directly within a given directory (non-recursive).

    Args:
        directory_path (str): The path to the directory to search.

    Returns:
        list: A list of absolute paths to files found in the specified directory.
    """
    files_only = []
    for entry in os.listdir(directory_path):
        full_path = os.path.join(directory_path, entry)
        if os.path.isfile(full_path):
            files_only.append(full_path)
    return files_only

''' Finds indices in a list that meet a specific condition. '''
def find_indices(list_obj, condition):
    return [index for index, element in enumerate(list_obj) if condition(element)]

''' Ensures filename ends with the right extenion and exists.'''
def check_filename(filename):
    if not filename.endswith(".npz"):
        print("Invalid file extension, exiting.")
        return False
    if not os.path.isfile(filename):
        print("This file does not exist")
        return False
    return True

def scale_image(image, min_temp, max_temp):
    # Clip data above/below our target range
    image[image < min_temp] = min_temp
    image[image > max_temp] = max_temp

    # Scale the image to the range [0, 1]
    scaled_image = (image - min_temp) / (max_temp - min_temp)

    # Scale the image to the desired range [min_value, max_value]
    scaled_image = scaled_image * 255

    # Turn data into 8-bit integers
    scaled_image = scaled_image.astype(np.uint8)
    return scaled_image

files = get_files_in_directory(folder_name)
print(files)
export_folder = folder_name + "classes/"
# Only write data to export folder if -e flag provided
#export_folder = filename[:-4] + "_export/"
if not os.path.exists(export_folder):
    os.makedirs(export_folder)

for c in classes:
    class_filename = export_folder + c + "/"
    if not os.path.exists(class_filename):
        os.makedirs(class_filename)

IMAGE_SIZE = (320, 240)
mask = cv2.imread("mask.jpg", 0)
_ret, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for MP4
out = cv2.VideoWriter('output.mp4', fourcc, 27.0, (IMAGE_SIZE[0], IMAGE_SIZE[0])) 

# Load the data
data = np.load("gale_varied_strength.npz")
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


Y_OFFSET = int((IMAGE_SIZE[0] - IMAGE_SIZE[1])/2)

mask_rect = mask[Y_OFFSET:IMAGE_SIZE[1]+Y_OFFSET, :]

avg = None
exhaling = False

# Iterate through every frame
for i, img in enumerate(frames):
    
    # Mask the image
    img = cv2.bitwise_and(img, img, mask=mask_rect)

    # Get the masked values
    img_m = img[np.where(mask_rect==255)]

    # Subtract the min value from the image and masked values
    img -= np.min(img_m)
    img_m -= np.min(img_m)

    max_val = np.max(img_m)
    print(max_val)
    
    # Scale image
    scaled_orig = scale_image(img, 0, 12)


    scaled_orig[np.where(mask_rect==0)] = 0
    
    # Create square images
    scaled_orig_square = np.zeros((IMAGE_SIZE[0], IMAGE_SIZE[0]), dtype=np.uint8)
    scaled_orig_square[Y_OFFSET:IMAGE_SIZE[1]+Y_OFFSET, :] = scaled_orig
    scaled_orig_square = cv2.merge((scaled_orig_square, scaled_orig_square, scaled_orig_square))

    out.write(scaled_orig_square) # Write the frame to the output video

out.release()