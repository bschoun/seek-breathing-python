# Seek Breathing Python Scripts

This project has Python scripts to record and process thermal images from a Seek Thermal camera using the Thin Thermal Medium setup, and used to develop the thermal imaging-based VR breathing sensor for VR used in the [Breathing Lab project](https://github.com/bschoun/breathing-thermal-thin-medium).

I'm in the process of organizing and improving this repository, such as removing hard-coded references to data files and making a list of Python package requirements. My apologies and please bear with me. Feel free to reach out if you're interested in breathing data I've already collected.

## Scripts
[`record_npz.py`](record_npz.py) - Records breathing data in a .npz file. You will need to install the Seek Thermal SDK first, which can be found in the [Seek Developer Portal](https://developer.thermal.com/). The script will prompt you to inhale and exhale over the course of 2 minutes.

[`process_data.py`](process_data.py) - Image processing algorithm to detect exhales in a given .npz file. 

[`export_video.py`](export_video.py) - Exports a video from a .npz file, after applying pre-processing.

[`export_data.py`](export_data.py) - Exports .npz contents as image files. This is something I used to create images that I could feed into a CNN.
