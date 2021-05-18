**This sample provides reference for you to learn the Ascend AI Software Stack and cannot be used for commercial purposes.**

**This sample works with CANN 3.0.0 and later versions, and supports Atlas 200 DK and Atlas 300.**

**This readme file provides only guidance for running the sample in the command line. For details about how to run the sample in MindStudio, see [Running Image Samples in MindStudio](https://gitee.com/ascend/samples/wikis/Running%20Image%20Samples%20in%20MindStudio?sort_id=3736297).**

## Sample of Multi-Object Tracking with Object_Tracking_Video
Function: tracks multiple pedestrians in a scene with the **mot_v2.om** model.

Input: a crowd image

Output: an image with bounding box and ID for each person in the scene

### Prerequisites

Before deploying this sample, ensure that:

- The environment has been set up by referring to [Environment Preparation and Dependency Installation](https://gitee.com/ascend/samples/blob/master/python/environment/README.md).
- The development environment and operating environment of the corresponding product have been set up.

### Software Preparation

#### 1. Obtain the source package.

  You can download the source code in either of the following ways:

   - Command line (The download takes a long time, but the procedure is simple.)

     In the development environment, run the following commands as a non-root user to download the source repository:
        ```
     cd $HOME
     git clone https://gitee.com/ascend/samples.git
        ```
   - Compressed package (The download takes a short time, but the procedure is complex.)

     1. Click **Clone or download** in the upper right corner of the samples repository and click **Download ZIP**.

     2. Upload the .zip package to the home directory of a common user in the development environment, for example, **$HOME/ascend-samples-master.zip**.

     3. In the development environment, run the following commands to unzip the package:

      ```
     cd $HOME
     unzip ascend-samples-master.zip
      ```
#### 2. Obtain the model required by the application.

   Ensure you are in the project directory (`object_tracking_video/`) and run one of the following commands in the table to obtain the pedestrian tracking model used in the application.

	cd $HOME/samples/python/contrib/object_tracking_video/

| **Model**  |  **How to Obtain** |
| ---------- |  ----------------- |
| mot_v2.om | `wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1RU1UBVH5EBbVV4CVAPuNokSzpfx9A3Ug' -O model/mot_v2.om`  |
| mot_v2.onnx | `wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1Esjf7Mj-CTh-VQGNHEcpX2uwJlQEnrJD' -O model/mot_v2.onnx`  |

   ![Icon-note.gif](https://images.gitee.com/uploads/images/2020/1106/160652_6146f6a4_5395865.gif) **NOTE**
   >- `mot_v2.om` offline model you can use out-of-the-box without model conversion. If you use this then you can skip the next step on model conversion.
   >- `mot_v2.onnx` ONNX model for those that want to configure the model conversion process.
   
   From the project directory, navigate to the `scripts/` directory and run `get_sample_data.sh` to download sample images for testing the application later in section 4. The sample images will be saved in `data/`.
   
   ```
   cd $HOME/samples/python/contrib/object_tracking_video/script/
   bash get_sample_data.sh
   ```
#### 3. Convert the original model to a DaVinci model.

   **Note: Ensure that the environment variables have been configured in [Environment Preparation and Dependency Installation](https://gitee.com/ascend/samples/tree/master/python/environment).**

   1. Set the ***LD_LIBRARY_PATH*** environment variable.

      The ***LD_LIBRARY_PATH*** environment variable conflicts with the sample when Ascend Tensor Compiler (ATC) is used. Therefore, you need to set this environment variable separately in the command line to facilitate modification.
      
          export LD_LIBRARY_PATH=${install_path}/atc/lib64

   For **CANN 3.3.0-alpha006**: <br/>

   2. Go to the project directory (object_tracking_video) and run the model conversion command to convert the model:

          atc --input_shape="input.1:1,3,608,1088" --check_report=./network_analysis.report --input_format=NCHW --output=model/mot_v2 --soc_version=Ascend310 --framework=5 --model=model/mot_v2.onnx

#### 4. Obtain the test image required by the sample.

Navigate to the following project directory and then run the get data script to download test images.

    cd $HOME/samples/python/contrib/object_tracking_video/get_sample_data.sh


### Sample Running

**Note: If the development environment and operating environment are set up on the same server, skip step 1 and go to step 2 directly.**

1. Run the following commands to upload the **object_tracking_video** directory in the development environment to any directory in the operating environment, for example, **/home/HwHiAiUser**, and log in to the operating environment (host) as the running user (**HwHiAiUser**):
      ```
      scp -r $HOME/samples/python/contrib/object_tracking_video/  HwHiAiUser@xxx.xxx.xxx.xxx:/home/HwHiAiUser
      scp -r $HOME/samples/python/common/atlas_utils/   HwHiAiUser@xxx.xxx.xxx.xxx:/home/HwHiAiUser
      ssh HwHiAiUser@xxx.xxx.xxx.xxx
      ```

   ![Icon-note.gif](https://images.gitee.com/uploads/images/2020/1106/160652_6146f6a4_5395865.gif) **NOTE**

   > - Replace ***xxx.xxx.xxx.xxx*** with the IP address of the operating environment. The IP address of Atlas 200 DK is **192.168.1.2** when it is connected over the USB port, and that of Atlas 300 is the corresponding public network IP address.


2. Run the executable file.

   - If the development environment and operating environment are set up on the same server, run the following commands to set the operating environment variable and switch the directory:

     ```
     export LD_LIBRARY_PATH=
     source ~/.bashrc
     cd $HOME/samples/python/contrib/object_tracking_video/src
     python3 main.py ../data/
     ```

   - If the development environment and operating environment are set up on separate servers, run the following command to switch the directory:

     ```
     cd $HOME/object_tracking_video/src
     ```
     Run the following command to run the sample:
     ```
     python3 main.py 
     ```

### Result Checking


After the execution is complete, find the JPG image the inference results in `object_tracking_video/src/output/`.


<!-- Pedestrian Detection and Tracking on Atlas 200DK, a dlav0 version of [FairMOT](https://github.com/ifzhang/FairMOT).

## Introduction
Multi Object Tracking (MOT) is a challenging topic as it often has two seperate tasks for detection and tracking. Recent attention focus on accomplishing the two tasks in a single network to improve the inference speed. [FairMOT](https://github.com/ifzhang/FairMOT), compared to [JDE](https://github.com/Zhongdao/Towards-Realtime-MOT), uses anchor-free CenterNet as the backbone to balance the detection and re-id branches and Kalman Filter for bounding box state prediction, resulting state-of-the-arts accuracy and near real-time speed (30 fps) using good GPUs. The dlav0 version has slightly lower accuracy but ~2x faster. The speed on Atlas 200DK is ~8 FPS depending on number of detections.

## Tracking performance
### Sample Comparison for Unseen Video
<img src="assets/london_compare.gif" width="1000"/> 
Or <a href="https://www.youtube.com/watch?v=ndSdGqUV0cg">Youtube</a>

### Quantitative Comparison on [MOT Challenge](https://motchallenge.net/) using GTX1080
<img src="assets/quantitative_compare.png" width="400"/> 

### Important Notes
As the tracking/association part uses CPU and cannot be benefitted by HPU, the number of detection impacts the speed a lot.

## Installation
Python 3.6.9
### Download Model
```
cd model
./download.sh
cd ..
```

### Install Dependencies
```
pip3 install -r requirements.txt
```

### Run
```
python3 main.py --input_video "\Path to video"
```

### Acknowledgement
A large part of the code is borrowed from [FairMOT](https://github.com/ifzhang/FairMOT), [JDE](https://github.com/Zhongdao/Towards-Realtime-MOT), and [CenterNet](https://github.com/xingyizhou/CenterNet). Thanks for their wonderful works.
