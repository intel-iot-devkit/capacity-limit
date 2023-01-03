# DISCONTINUATION OF PROJECT #
This project will no longer be maintained by Intel.
This project has been identified as having known security escapes.
Intel has ceased development and contributions including, but not limited to, maintenance, bug fixes, new releases, or updates, to this project.
Intel no longer accepts patches to this project.
# Retail Pandemic Reference Implementation - Capacity Limit

| Details               |              |
|-----------------------|---------------|
| Target OS:            |  Ubuntu\* 18.04 LTS   |
| Programming Language: |  Python* 3.5 |
| Time to Complete:     |  1 hour     |

![linecrossing](./doc/images/capacity-limit.png)

## What it does

This reference implementation showcases a retail capacity limit application where a people counter keeps track of how many people are in the store based on the detected people entering and exiting the store. Virtual lines are configured in entrance and exit areas to indicate the limits of entering and exiting the area of interest.

## Requirements

### Hardware

* 6th gen or greater Intel® Core™ processors or Intel® Xeon® processor, with 8Gb of RAM

### Software

* [Ubuntu 18.04](http://releases.ubuntu.com/18.04/)

* [Intel® Distribution of OpenVINO™ toolkit 2020.3 Release](https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit.html)

## How It works

The application uses the Inference Engine and Model Downloader included in the Intel® Distribution of OpenVINO Toolkit doing the following steps:

1. Ingests video from a file, processing frame by frame.
2. Detects people in the frame using a pre-trained DNN model.
3. Extracts features from people detected using a second pre-trained DNN in order to do a tracking of the people.
4. Check if people cross the predefined virtual gates based on the coordinates and identify in which direction the virtual gate was closed to determine if it is an entry or exit event.
5. People counter gets updated based on the entry and exit data.

The DNN models are optimized for Intel® Architecture and are included with Intel® Distribution of OpenVINO™ toolkit.

![architecture image](./doc/images/flow-capacity-limit.png)

## Setup

### Get the code

Clone the reference implementation:

```bash
sudo apt-get update && sudo apt-get install git
git clone github.com:intel-iot-devkit/capacity-limit.git
```

### Install Intel® Distribution of OpenVINO™ Toolkit

Refer to https://software.intel.com/en-us/articles/OpenVINO-Install-Linux for more information about how to install and setup the Intel® Distribution of OpenVINO™ toolkit.

### Installing the requirements

To install the dependencies of the Reference Implementation, run the following commands:

```bash
  cd <path_to_capacity-limit-directory>
  pip3 install -r requirements.txt
```

### Which model to use

This application uses the [person-detection-retail-0013](https://docs.openvinotoolkit.org/2020.3/_models_intel_person_detection_retail_0013_description_person_detection_retail_0013.html) and [person-reidentification-retail-0300](https://docs.openvinotoolkit.org/2020.3/_models_intel_person_reidentification_retail_0300_description_person_reidentification_retail_0300.html) Intel® pre-trained models, that can be downloaded using the **model downloader**. The **model downloader** downloads the __.xml__ and __.bin__ files that is used by the application.

To install the dependencies of the RI and to download the models Intel® model, run the following command:

```bash
mkdir models
cd models
python3 /opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader/downloader.py --name person-detection-retail-0013 --precisions FP32
python3 /opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader/downloader.py --name person-reidentification-retail-0300 --precisions FP32
```

The models will be downloaded inside the following directories:

```bash
- models/intel/person-detection-retail-0013/FP32/
- models/intel/person-reidentification-retail-0300/FP32/
```

### The Config File

The _config.json_ contains the path to the videos and models that will be used by the application and also the coordinates of a virtual line.

The _config.json_ file is of the form name/value pair. Find below an example of the _config.json_file:

```bash
{
  "video": "path/to/video/myvideo.mp4",
  "pedestrian_model_weights": "models/intel/person-detection-retail-0013/FP32/person-detection-retail-0013.bin",
  "pedestrian_model_description": "models/intel/person-detection-retail-0013/FP32/person-detection-retail-0013.xml",
  "reidentification_model_weights": "models/intel/person-reidentification-retail-0300/FP32/person-reidentification-retail-0300.bin",
  "reidentification_model_description": "models/intel/person-reidentification-retail-0300/FP32/person-reidentification-retail-0300.xml",
  "coords": [[50, 45], [80, 45]]
}
```

Note that __coords__ represents a virtual gate defined by two points with x,y coordinates.

### Which Input video to use

The application works with any input video format supported by [OpenCV](https://opencv.org/).

Sample video: https://www.pexels.com/video/black-and-white-video-of-people-853889/

Data set subject to license https://www.pexels.com/license. The terms and conditions of the data set license apply. Intel does not grant any rights to the data files.

To use any other video, specify the path in config.json file.

## Setup the environment

You must configure the environment to use the Intel® Distribution of OpenVINO™ toolkit one time per session by running the following command:

```bash
source /opt/intel/openvino/bin/setupvars.sh -pyver 3.5
```

__Note__: This command needs to be executed only once in the terminal where the application will be executed. If the terminal is closed, the command needs to be executed again.

## Run the application

Change the current directory to the project location on your system:

```bash
    cd <path-to-capacity-limit-directory>
```

Run the python script.

```bash
python3 capacitylimit.py
```
