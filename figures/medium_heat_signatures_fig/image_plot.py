import matplotlib.pyplot as plt
import numpy as np
import cv2
import os.path

categories = ["center_gale", "left_gale", "right_gale", "waft", "calm_mouth", "nose"]

fig, axes = plt.subplots(nrows=len(categories), ncols=3, figsize=(5.5,8))

# Disable the coordinate display
for cidx, c in enumerate(categories):
    for i in range(3):
        img = cv2.imread(c + "/" + str(i+1) + ".png")
        axes[cidx, i].set_xticklabels([])
        axes[cidx, i].set_yticklabels([])
        axes[cidx, i].imshow(img)
        # Hide x-axis tick marks (bottom and top)
        axes[cidx, i].tick_params(axis='x', which='both', bottom=False, top=False)

        # Hide y-axis tick marks (left and right)
        axes[cidx, i].tick_params(axis='y', which='both', left=False, right=False)
        #axes[cidx, i].xaxis.set_visible(False)
        #axes[cidx, i].yaxis.set_visible(False)
        if i == 0:
            axes[cidx, i].set_ylabel(c)

fig.suptitle("Medium Heat Signatures")
plt.tight_layout()
plt.show()