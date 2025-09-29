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

for f in files:

    # Load the data
    data = np.load(f)
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


    y_offset = int((IMAGE_SIZE[0] - IMAGE_SIZE[1])/2)

    mask_rect = mask[y_offset:IMAGE_SIZE[1]+y_offset, :]

    avg = None
    exhaling = False

    # Iterate through every frame
    for i, img in enumerate(frames):
        
        # Mask the image
        img = cv2.bitwise_and(img, img, mask=mask_rect)

        # Get the masked values
        img_m = img[np.where(mask_rect==255)]


        
        scaled_img = scale_image(img, np.min(img_m), np.max(img_m))
        scaled_img[np.where(mask_rect==0)] = 0

        # Create square images
        scaled_img_square = np.zeros((IMAGE_SIZE[0], IMAGE_SIZE[0]), dtype=np.uint8)
        scaled_img_square[y_offset:IMAGE_SIZE[1]+y_offset, :] = scaled_img

        img -= np.min(img_m)
        img_m -= np.min(img_m)

        '''if exhaling_data[i]:
            if "gale" in f:
                cv2.imwrite(export_folder + "gale/" + str(gale_count) + ".jpg", scaled_img_square)
                gale_count += 1
            elif "waft" in f:
                cv2.imwrite(export_folder + "waft/" + str(waft_count) + ".jpg", scaled_img_square)
                waft_count += 1
            else:
                cv2.imwrite(export_folder + "other/" + str(other_count) + ".jpg", scaled_img_square)
                other_count += 1
        else:
            cv2.imwrite(export_folder + "none/" + str(none_count) + ".jpg", scaled_img_square)
            none_count += 1'''


        # Get the masked values
        
        if avg is None:
            avg = img.copy()
        else:
            diff = img-avg
            diff_m = diff[np.where(mask_rect==255)]
            scaled_diff = scale_image(diff, -3, 5)
            scaled_diff[np.where(mask_rect==0)] = 0

            scaled_diff_square = np.zeros((IMAGE_SIZE[0], IMAGE_SIZE[0]), dtype=np.uint8)
            scaled_diff_square[y_offset:IMAGE_SIZE[1]+y_offset, :] = scaled_diff
            
            midrange_d_val = (np.max(diff_m) + np.min(diff_m))/2.0
                
            # Deterine exhaling or not exhaling
            if not exhaling and midrange_d_val > 0.5:
                exhaling = True
            elif exhaling and midrange_d_val <= 0.0:
                exhaling = False

            avg = cv2.accumulateWeighted(img, avg, 0.01)
        
            if exhaling and exhaling_data[i] == True:
                if "gale" in f:
                    cv2.imwrite(export_folder + "gale/" + str(gale_count) + ".jpg", scaled_img_square)
                    gale_count += 1
                elif "waft" in f:
                    cv2.imwrite(export_folder + "waft/" + str(waft_count) + ".jpg", scaled_img_square)
                    waft_count += 1
                else:
                    cv2.imwrite(export_folder + "other/" + str(other_count) + ".jpg", scaled_img_square)
                    other_count += 1
            else:
                cv2.imwrite(export_folder + "none/" + str(none_count) + ".jpg", scaled_img_square)
                none_count += 1