# data from https://allisonhorst.github.io/palmerpenguins/

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.size': 24}) # Sets the default font size to 14

detection_method = ("Statistics", "CNN")
exhale_types = {
    'Gale': (0.49, 0.26),
    'Waft': (0.18, 0.11),
}

x = np.arange(len(detection_method))  # the label locations
width = 0.4  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(layout='constrained', figsize=(8, 8))

for attribute, measurement in exhale_types.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute)
    ax.bar_label(rects, padding=3)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Delay (seconds)')
ax.set_title('Statistical vs CNN Exhale Ends')
ax.set_xticks(x + width*.5, detection_method)
ax.legend(loc='upper left', ncols=3)
ax.set_ylim(0, 0.6)

plt.show()