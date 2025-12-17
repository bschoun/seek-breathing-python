import numpy as np
import cv2
import os.path
import argparse
import matplotlib.pyplot as plt
import tensorflow as tf
from enum import Enum
import math



#folder = "data/data_final/"
folder = "./"
#files = ["center_gale1", "center_gale1", "center_gale3", "center_gale4", "left_gale1", "right_gale1", "left_gale2", "right_gale2","left_gale3", "right_gale3"]#, "directional_gale1", "directional_gale2"]
files = ["baseline"]
start_delays = {"gale":np.array([]), "waft":np.array([]), "calm_mouth":np.array([]), "nose":np.array([])}
end_delays = {"gale":np.array([]), "waft":np.array([]), "calm_mouth":np.array([]), "nose":np.array([])}


extension = ".npz"
cnn_early_end = True
plot_exhale_cues = True
plot_exhales_detected = False
plot_identified_gestures = False
plot_directional_gale = True
plot_delay = False
if plot_delay:
    plt.rcParams.update({'font.size': 24})
plot_breathing = True
show_video = False

gale_left_accurate = 0
gale_left_false = 0
gale_left_total = 0
gale_right_accurate = 0
gale_right_false = 0
gale_right_total = 0
gale_center_accurate = 0
gale_center_false = 0
gale_center_total = 0

ROWS = math.ceil(len(files)/2.0)
COLS = 2

class ExhaleType(Enum):
    GALE = 0
    WAFT = 1
    NONE = 2

class DirectionType(Enum):
    NONE = 0
    LEFT = 1
    CENTER = 2
    RIGHT = 3

#debounce = 1


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


def classify_image(_image):

    image = cv2.resize(_image, (224, 224), interpolation=cv2.INTER_AREA)

    # Make the image a numpy array and reshape it to the models input shape.
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize the image array
    image = (image / 127.5) - 1

    #image = np.asarray(image, dtype=np.uint8).reshape(1, 224, 224, 3)

    # Predict
    interpreter.set_tensor(input_details[0]['index'], image)

    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]
    return output_data

mask = cv2.imread("mask.jpg", 0)
_ret, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

f = (7,1.5+1.75*ROWS)
if ROWS == 1:
    f = (7,3)
elif ROWS == 2:
    f = (7,6)
elif ROWS == 5:
    f = (7,10)

# Create a figure with a 2x2 grid of subplots
fig, axes = plt.subplots(nrows=ROWS, ncols=COLS, figsize=f, sharex=True, sharey=True)
handles = []

interpreter = None
input_details = None
output_details = None

# Load TFLite model
class_names = ["Gale", "Waft", "None"]

GALE_CONFIDENCE = 0.99
NONE_CONFIDENCE = 0.95
WAFT_CONFIDENCE = 0.99
    
# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

interpreter = tf.lite.Interpreter(model_path="model/converted_tflite/model_unquant.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']

''' CONSTANTS '''

FPS = 27
#START_STD_DEV = 2.5
#END_STD_DEV = 1.5
EXHALE_TIMEOUT = 7
HISTORY_SECONDS = 1
EXHALE_START_THRESHOLD = 0.5
EXHALE_END_THRESHOLD = -0.25
GALE_THRESHOLD = -0.25
WAFT_THRESHOLD = 2.0
IMAGE_SIZE = (320, 240)
GALE_LEFT_X = 130
GALE_RIGHT_X = 190


Y_OFFSET = int((IMAGE_SIZE[0] - IMAGE_SIZE[1])/2)


direction = DirectionType.NONE

mask_rect = mask[Y_OFFSET:IMAGE_SIZE[1]+Y_OFFSET, :]
wait_ms = 1

history_alpha = 1/(HISTORY_SECONDS*FPS)
#short_history_alpha = 1/(SHORT_HISTORY*FPS)

exhale_type = ExhaleType.NONE
exhaling = False

classify = True

# Load the recorded data
for fidx, f in enumerate(files):

    # Reset exhale_type
    exhale_type = ExhaleType.NONE
    exhaling = False
    direction = DirectionType.NONE

    exhale_actual_starts = []
    exhale_actual_ends = []

    print("loading data %s" % f)
    # Load the data
    data = np.load(folder + f + extension)
    frames = data["frames"]
    timestamps = data["timestamps"]
    timestamps -= timestamps[0]     # Subtract the intital time stamp from all timestamps, converting to relative time

    exhaling_data = None
    if "exhaling" in data:
        exhaling_data = data["exhaling"]

    #print(exhaling_data)
        
    # Make sure the timestamps and frames are the same length
    if len(timestamps) > len(frames):
        timestamps = timestamps[:len(frames)]

    # Trim exhaling if needed
    if exhaling_data is not None:
        if len(exhaling_data) > len(frames):
            exhaling_data = exhaling_data[:len(frames)]

    moving_avg = None # 

    # Diff (historical) values
    midranges = []
    iqr_max_diffs = []

    # Singular image values
    max_values = []

    # Exhales and gestures
    exhale_starts = []
    exhale_ends = []

    
    gesture_gale_starts = []
    gesture_gale_left_starts = []
    gesture_gale_center_starts = []
    gesture_gale_right_starts = []
    gesture_waft_starts = []
    gesture_gale_ends = []
    gesture_gale_left_ends = []
    gesture_gale_center_ends = []
    gesture_gale_right_ends = []
    gesture_waft_ends = []
    gesture_starts = []
    gesture_ends = []

    uppers = []

    initialized = False

    #exhale_end_threshold = 0
    #prev_max = 0

    # Iterate through every frame
    for i, img in enumerate(frames):

        # Mask the image
        img = cv2.bitwise_and(img, img, mask=mask_rect)

        # Get the masked values
        img_m = img[np.where(mask_rect==255)]

        # Subtract the min value from the image and masked values
        img -= np.min(img_m)
        img_m -= np.min(img_m)
        
        # Scale image
        scaled_orig = scale_image(img, 0, 10)
        scaled_orig[np.where(mask_rect==0)] = 0
        
        # Create square images
        scaled_orig_square = np.zeros((IMAGE_SIZE[0], IMAGE_SIZE[0]), dtype=np.uint8)
        scaled_orig_square[Y_OFFSET:IMAGE_SIZE[1]+Y_OFFSET, :] = scaled_orig
        scaled_orig_square = cv2.merge((scaled_orig_square, scaled_orig_square, scaled_orig_square))

        if not initialized:
            moving_avg = img.copy()
            initialized = True

        max_val = np.max(img_m)
            
        diff = img-moving_avg
        diff_m = diff[np.where(mask_rect==255)]
        

                    
        # Calculate midrange
        midrange = (np.max(diff_m) + np.min(diff_m))/2.0

        # IQR of the diff image
        q1, q3 = np.percentile(diff_m, [25, 75])
        iqr = q3 - q1
        upper = q3 + (1.5*(iqr))
        iqr_max_diff = upper - midrange
        
        
        cnn_end = False
        if cnn_early_end and exhaling and exhale_type is not ExhaleType.NONE:
            result = classify_image(scaled_orig_square)
            # Get the maximum value of the data
            m = np.argmax(result)
            if result[m] > NONE_CONFIDENCE and class_names[m] == "None":
                print("CNN END")
                cnn_end = True

        # Deterine exhaling or not exhaling
        if not exhaling:
            if midrange > EXHALE_START_THRESHOLD:
                exhaling = True
                exhale_starts.append(i)

        if exhaling:

            # Check for the start of an exhale
            if exhale_type == ExhaleType.NONE:
                if iqr_max_diff < GALE_THRESHOLD:
                    exhale_type = ExhaleType.GALE
                    gesture_gale_starts.append(i)
                elif iqr_max_diff > WAFT_THRESHOLD:
                    exhale_type = ExhaleType.WAFT
                    gesture_waft_starts.append(i)

            if exhale_type == ExhaleType.GALE:
                # Get x pos
                (_min, _max, _min_loc, _max_loc) = cv2.minMaxLoc(img)
                if "left" in f:
                    gale_left_total += 1
                if "right" in f:
                    gale_right_total += 1
                if "center" in f:
                    gale_center_total += 1
                
                if _max_loc[0] <= GALE_LEFT_X:
                    if "left" in f:
                        gale_left_accurate += 1
                    else:
                        gale_left_false += 1
                    if direction != DirectionType.LEFT:
                        if direction == DirectionType.RIGHT:
                            gesture_gale_right_ends.append(i)
                        elif direction == DirectionType.CENTER:
                            gesture_gale_center_ends.append(i)
                        direction = DirectionType.LEFT
                        gesture_gale_left_starts.append(i)
                elif _max_loc[0] >= GALE_RIGHT_X:
                    if "right" in f:
                        gale_right_accurate += 1
                    else:
                        gale_right_false += 1
                    if direction != DirectionType.RIGHT:
                        if direction == DirectionType.LEFT:
                            gesture_gale_left_ends.append(i)
                        elif direction == DirectionType.CENTER:
                            gesture_gale_center_ends.append(i)
                        direction = DirectionType.RIGHT
                        gesture_gale_right_starts.append(i)
                else:
                    if "center" in f:
                        gale_center_accurate += 1
                    else:
                        gale_center_false += 1
                    if direction != DirectionType.CENTER:
                        if direction == DirectionType.LEFT:
                            gesture_gale_left_ends.append(i)
                        elif direction == DirectionType.RIGHT:
                            gesture_gale_right_ends.append(i)
                        direction = DirectionType.CENTER
                        gesture_gale_center_starts.append(i)
                    
            # Check for the end of an exhale
            if midrange <= EXHALE_END_THRESHOLD or timestamps[i] - timestamps[exhale_starts[-1]] > EXHALE_TIMEOUT or cnn_end:
                exhaling = False
                
                exhale_ends.append(i)
                if exhale_type == ExhaleType.WAFT:
                    gesture_waft_ends.append(i)
                elif exhale_type == ExhaleType.GALE:
                    gesture_gale_ends.append(i)
                    if direction == DirectionType.RIGHT:
                        gesture_gale_right_ends.append(i)
                    elif direction == DirectionType.LEFT:
                        gesture_gale_left_ends.append(i)
                    elif direction == DirectionType.CENTER:
                        gesture_gale_center_ends.append(i)
                direction = DirectionType.NONE
                exhale_type = ExhaleType.NONE

        moving_avg = cv2.accumulateWeighted(img, moving_avg, 1/(HISTORY_SECONDS*27))

        if show_video:

            cv2.imshow("Original", scaled_orig)
            cv2.imshow("Moving average", scale_image(moving_avg, 0, 10))
            cv2.imshow("Diff", scale_image(img-moving_avg, -5, 5))

            # Exit on pressing 'q'
            if cv2.waitKey(wait_ms) & 0xFF == ord('q'): # the 'q' key to break the loop
                break
        
        # Store data for plotting
        if i > 0:
            if exhaling_data[i] != exhaling_data[i-1]:
                if exhaling_data[i] == True:
                    exhale_actual_starts.append(i)
                else:
                    exhale_actual_ends.append(i)
        
        iqr_max_diffs.append(iqr_max_diff)
        uppers.append(upper)

        midranges.append(midrange)
        max_values.append(max_val)


    if len(exhale_ends) < len(exhale_starts):
        exhale_ends.append(i)
    if len(exhale_actual_ends) < len(exhale_actual_starts):
        exhale_actual_ends.append(i)
    if len(gesture_gale_ends) < len(gesture_gale_starts):
        gesture_gale_ends.append(i)
    if len(gesture_waft_ends) < len(gesture_waft_starts):
        gesture_waft_ends.append(i)
    if len(gesture_gale_left_ends) < len(gesture_gale_left_starts):
        gesture_gale_left_ends.append(i)
    if len(gesture_gale_right_ends) < len(gesture_gale_right_starts):
        gesture_gale_right_ends.append(i)
    if len(gesture_gale_center_ends) < len(gesture_gale_center_starts):
        gesture_gale_center_ends.append(i)


    t_cue_end_indices = np.array([timestamps[j] for j in exhale_actual_ends])
    t_cue_start_indices = np.array([timestamps[j] for j in exhale_actual_starts])

    if len(exhale_starts) > len(exhale_actual_starts):
        exhale_start_tmp = []
        exhale_end_tmp = []
    
        print("iterating")
        for tidx, t in enumerate(exhale_starts):
            if tidx == 0:
                exhale_start_tmp.append(t)
                exhale_end_tmp.append(exhale_ends[tidx])
            else:
                if exhale_starts[tidx] - exhale_start_tmp[-1] > 27 and (exhale_ends[tidx] - exhale_starts[tidx] > 27*2 or tidx == len(exhale_starts)-1):
                    exhale_start_tmp.append(t)
                    exhale_end_tmp.append(exhale_ends[tidx])
        
        exhale_starts = exhale_start_tmp
        exhale_ends = exhale_end_tmp


    t_end_indices = np.array([timestamps[j] for j in exhale_ends])
    t_start_indices = np.array([timestamps[j] for j in exhale_starts])

    print(t_end_indices)
    print(t_start_indices)
    print(t_cue_end_indices)
    print(t_cue_start_indices)


    category = 'gale'
    if 'waft' in f:
        category = 'waft'
    elif 'nose' in f:
        category = 'nose'
    elif 'mouth' in f:
        category = 'calm_mouth'


    #start_delays[category] = np.append(start_delays[category], t_start_indices - t_cue_start_indices)
    #end_delays[category] = np.append(end_delays[category], t_end_indices[:-1] - t_cue_end_indices[:-1])




    # Draw a horizontal black line at y=0
    x = int(fidx/COLS)
    y = fidx%COLS
    a = 0.25
    if ROWS == 1:
        ax = axes[y]
    elif COLS == 1:
        ax = axes[x]
    else:
        ax = axes[x,y]
    # Plot the exhale starts and ends
    t_end_indices = [timestamps[j] for j in exhale_actual_ends]
    t_start_indices = [timestamps[j] for j in exhale_actual_starts]
    
    if plot_exhale_cues:
        for j, t in enumerate(exhale_actual_starts):
            c = '#888888'
            #if exhale_types[j] == 'W':
            #    c = 'cyan'
            if (j==0):
                ax.axvspan(t_start_indices[j], t_end_indices[j], color=c, alpha=a, hatch=r'\\\\', label="exhale cue")
            else:
                ax.axvspan(t_start_indices[j], t_end_indices[j], color=c, alpha=a, hatch=r'\\\\')

    if plot_exhales_detected:
        t_end_indices = [timestamps[j] for j in exhale_ends]
        t_start_indices = [timestamps[j] for j in exhale_starts]
        for j, t in enumerate(exhale_starts):
            c = "#00FF00"
            #if exhale_types[j] == 'W':
            #    c = 'cyan'
            if (j==0):
                ax.axvspan(t_start_indices[j], t_end_indices[j], color=c, hatch=r'////', alpha=a, label="detected exhale")
            else:
                ax.axvspan(t_start_indices[j], t_end_indices[j], color=c, hatch=r'////', alpha=a)

    if plot_identified_gestures:
        # Plot the gesture starts and ends
        t_end_indices = [timestamps[j] for j in gesture_gale_ends]
        t_start_indices = [timestamps[j] for j in gesture_gale_starts]
        for j, t in enumerate(gesture_gale_starts):
            c = "#00FFFF"
            #if exhale_types[j] == 'W':
            #    c = 'cyan'
            if (j==0):
                ax.axvspan(t_start_indices[j], t_end_indices[j], color=c, hatch=r'////', alpha=a, label="gale")
            else:
                ax.axvspan(t_start_indices[j], t_end_indices[j], color=c, hatch=r'////', alpha=a)

        # Plot the gesture starts and ends
        t_end_indices = [timestamps[j] for j in gesture_waft_ends]
        t_start_indices = [timestamps[j] for j in gesture_waft_starts]
        for j, t in enumerate(gesture_waft_starts):
            c = "#FF00FF"
            if j == 0:
                ax.axvspan(t_start_indices[j], t_end_indices[j], hatch=r'////', color=c, alpha=a, label="waft")
            else:
                ax.axvspan(t_start_indices[j], t_end_indices[j], hatch=r'////', color=c, alpha=a)

    if plot_directional_gale:
        # Plot the gesture starts and ends
        t_end_indices = [timestamps[j] for j in gesture_gale_left_ends]
        t_start_indices = [timestamps[j] for j in gesture_gale_left_starts]
        print(t_start_indices)
        print(t_end_indices)
        for j, t in enumerate(gesture_gale_left_starts):
            c = "#FF0000"
            #if exhale_types[j] == 'W':
            #    c = 'cyan'
            if (j==0):
                ax.axvspan(t_start_indices[j], t_end_indices[j], color=c, hatch=r'////', alpha=a, label="left")
            else:
                ax.axvspan(t_start_indices[j], t_end_indices[j], color=c, hatch=r'////', alpha=a)

        # Plot the gesture starts and ends
        t_end_indices = [timestamps[j] for j in gesture_gale_center_ends]
        t_start_indices = [timestamps[j] for j in gesture_gale_center_starts]
        
        for j, t in enumerate(gesture_gale_center_starts):
            c = "#00FF00"
            if (j==0):
                ax.axvspan(t_start_indices[j], t_end_indices[j], color=c, hatch=r'////', alpha=a, label="center")
            else:
                ax.axvspan(t_start_indices[j], t_end_indices[j], color=c, hatch=r'////', alpha=a)

        # Plot the gesture starts and ends
        t_end_indices = [timestamps[j] for j in gesture_gale_right_ends]
        t_start_indices = [timestamps[j] for j in gesture_gale_right_starts]
        for j, t in enumerate(gesture_gale_right_starts):
            c = "#0000FF"
            if (j==0):
                ax.axvspan(t_start_indices[j], t_end_indices[j], color=c, hatch=r'////', alpha=a, label="right")
            else:
                ax.axvspan(t_start_indices[j], t_end_indices[j], color=c, hatch=r'////', alpha=a)

    ax.axhline(0.0, color='black')
    
    
    handle, = ax.plot(timestamps, np.array(midranges), label='diff midrange')
    if plot_identified_gestures:
        handle, = ax.plot(timestamps, np.array(iqr_max_diffs), color='C2',  label='iqr upper - mid')

    # Display labels, show
    if y == 0:
        ax.set_ylabel("temperature (°C)")
    if x == ROWS - 1:
        ax.set_xlabel("time (seconds)")
    ax.set_title(f)
    if plot_identified_gestures:
        ax.set_ylim(-2, 5)
    elif plot_exhales_detected:
        ax.set_ylim(-2, 4)
    ax.set_ylim(-1,2)
    ax.set_xlim(0, 30)
    #if fidx == 1:
    #    axes[x, y].legend()
    #axes[x, y].grid()

if plot_breathing:

    title = ""

    if plot_identified_gestures:
        title += "Exhale Gesture Detection"
        if cnn_early_end:
            title += " (CNN early end)"

    elif plot_exhales_detected:
        title += "Exhale Detection" 
        if cnn_early_end:
            title += " (CNN early end)"

    elif plot_directional_gale:
        title += "Directional Gale"

    fig.suptitle(title)

    # Collect handles and labels from all relevant axes
    handles, labels = [], []
    for i, ax in enumerate(fig.get_axes()):
        h, l = ax.get_legend_handles_labels()
        for idx, label in enumerate(l):
            if not label in labels:
                labels.append(label)
                handles.append(h[idx])

    fig.legend(handles, labels, loc='upper center', ncols=5, bbox_to_anchor=(0, -0.06, 1, 1))

    if (plot_directional_gale):
        if gale_center_total > 0:
            print("GALE CENTER Accuracy: %.4f False positive: %.4f" % (gale_center_accurate/gale_center_total, gale_center_false/gale_center_total))
        else:
            print("GALE CENTER Accuracy: None detected")
        if gale_right_total > 0:
            print("GALE RIGHT Accuracy: %.4f False positive: %.4f" % (gale_right_accurate/gale_right_total, gale_right_false/gale_right_total))
        else:
            print("GALE RIGHT Accuracy: None detected")
        if gale_left_accurate > 0:
            print("GALE LEFT Accuracy: %.4f False positive: %.4f" % (gale_left_accurate/gale_left_total, gale_left_false/gale_left_total))
        else:
            print("GALE LEFT Accuracy: None detected")

    plt.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust layout to make space for the legend
    #plt.tight_layout()
    plt.show()

if plot_delay:

    delay_type = ("Exhale start", "Exhale end")
    exhale_means = {'gale':(round(np.mean(start_delays["gale"]),2), round(np.mean(end_delays["gale"]),2)),
                    'waft': (round(np.mean(start_delays["waft"]),2), round(np.mean(end_delays['waft']),2)),
                    'calm mouth': (round(np.mean(start_delays["calm_mouth"]),2), round(np.mean(end_delays['calm_mouth']),2)),
                    'nose': (round(np.mean(start_delays["nose"]),2), round(np.mean(end_delays['nose']),2))}

    x = np.arange(len(delay_type))  # the label locations
    width = 0.2  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained', figsize=(16, 8))

    for attribute, measurement in exhale_means.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=3)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Delay (seconds)')
    title = 'Exhale Detection Delay'
    if cnn_early_end:
        title += " (CNN early end)"
    else:
        title += " (statistics only)"
    ax.set_title(title)
    ax.set_xticks(x + width*1.5, delay_type)
    ax.legend(loc='upper left', ncols=4)
    ax.set_ylim(0, 0.6)
    plt.show()