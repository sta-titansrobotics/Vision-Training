# Object Detection API with TensorFlow 2 (for lemonlight)

## Requirements

[![Python 3.6](https://img.shields.io/badge/Python-3.6-3776AB)](https://www.python.org/downloads/release/python-360/)
[![TensorFlow 2.2](https://img.shields.io/badge/TensorFlow-2.2-FF6F00?logo=tensorflow)](https://github.com/tensorflow/tensorflow/releases/tag/v2.2.0)
[![Protobuf Compiler >= 3.0](https://img.shields.io/badge/ProtoBuf%20Compiler-%3E3.0-brightgreen)](https://grpc.io/docs/protoc-installation/#install-using-a-package-manager)

## Installation

You can install the TensorFlow Object Detection API with Python Package
Installer (pip) as well as protobuf

Clone the TensorFlow Models repository and proceed to one of the installation
options.

Alternatively, you can install the zip of the repo and unzip it to use as the main project directory.

```bash
#change this to working repo
git clone https://github.com/tensorflow/models.git
```

### Python Package Installation

#### Install Anaconda Prompt

If you do not already have it, install anaconda prompt from [here](https://www.anaconda.com/download/success) and open a new prompt window once installed

#### Create Conda Environment

You may have to change the directories to where you have the github repo installed if it is not in a standard location or if you installed it through the zip file.

```bash
#create environment
conda create -n tf2 python=3.9
#enter environment
conda activate tf2
#enter the installed github repo's folder (you may have to change)
cd models/research
```

#### Protobuf usage

Before you start using protobuf, you must check the version you have installed both in pip and in your conda environment. 

```bash
#check installed version on conda env (if this outputs a version other than 3.20.3, you must manually change the protoc version that anaconda uses)
protoc --version
#check installed version through pip (don't worry if this shows blank)
pip show protobuf
```

#### Changing Protobuf Version In Conda

install the correct version of protoc for conda from https://github.com/protocolbuffers/protobuf/releases/v3.20.3

use the command ```where protoc``` to see where the protoc installation is used and replace it with the manually installed version of 3.20.3 (in the zip) from the bin folder to wherever ```where protoc``` is telling you its installed and overwrite the current installed exe.

Double check versions are correct

```bash
#check installed version on conda env
#if this outputs a version other than 3.20.3, you did something wrong and must redo the previous step or find further help
protoc --version
#check installed version through pip (still, don't worry if this shows blank)
pip show protobuf
```

#### Compile Protos and Install Tensorflow Library
```bash
#compile protos
protoc object_detection/protos*.proto --python_out=.

#install the actual tensorflow library
python -m pip install .
```

### Test the installation

```bash
python object_detection/builders/model_builder_tf2_test.py
```

If the test fails with an error like ```cannot import name 'builder' from 'google.protobuf.internal'``` despite having the correct version of protoc/protobuf, you have to change the internal version (the pip installed version) to ```3.20.3``` from ```3.19.6``` by running these commands otherwise, leave the internal version on ```3.19.6```

```bash
pip uninstall protobuf
pip install protobuf==3.20.3
#check versions (both should say 3.20.3)
protoc --version
pip show protobuf
#run the installation test again
python object_detection/builders/model_builder_tf2_test.py
```

## Training A Model

### Organize and Label Files

Bring images into the ```images``` folder and sort them into either the ```train``` or ```validate``` subfolders.
If these folders do not exist yet, create them inside the ```images``` folder
```bash
cd object_detection/images
#copy your dataset/create the two needed subfolders in any way you want 
```

#### If your images are already labelled, you can skip this step

To label your images, you must install the labelimg pip library and use it to label your images as you desire

```bash
pip install labelimg
```

For documentation on how to install and use the labelimg library refer to here https://pypi.org/project/labelImg/

For the labelled images to be converted to something readable to tensorflow, you must convert the labels into two csv files
```bash
cd ..
python "csv converter.py"
```

To make the labels for both train and validate folders actually readable by tensorflow, you must convert them into tfrecord files

To begin, you must open the file ```generate_tfrecord.py``` and edit it so that you make the label names and numbers match the ones you made in labelimg

To generate these tfrecords, run the following commands
```bash
#you may have to change some of the directories to wherever the csv files were generated in the previous step
python generate_tfrecord.py --csv_input=images/labels/validate_labels.csv --image_dir=images/validate --output_path=validate.record
python generate_tfrecord.py --csv_input=images/labels/train_labels.csv --image_dir=images/train --output_path=train.record
```

Change the labelmap file in ```images/labels``` to have all the labels match those you have used in previous steps. You can see how to setup the syntax for different classes in the labelmap already configured.


#### Selecting and preparing your base model

Since we are using the tensorflow workflow to create an object detection model, we must first select a model from the [model zoo](object_detection/g3doc/tf2_detection_zoo.md)

Here you must select a specific base model to use as the foundation of the model and its training. For this use case, you must select a model with the outputs ```Boxes``` (only), and you may select your own out of these

For this tutorial, we will be using the [SSD MobileNet V2 FPNlite](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8.tar.gz) which is already downloaded in this library

#### Changing the model

If you wish to use a different model than the one used in this tutorial, you must download the ```'your selected model'.tar.gz``` from the model zoo.

Extract the compressed ```tar.gz``` file, and move its subfolder with the same name into the main working directory ```object_detection```

_If you wish to have an example of what the directory should look like when it is done, check the mobilnet640 or mobilnet320 directories_

#### Configure the model config

Enter the directory that you have set up for the model you selected (for those simply following the tutorial, that would be at ```mobilnet640```)

Here, you should find the file ```pipeline.config``` where you will have to edit a few parameters

Under ```train_config``` you should find the parameter ```batch_size``` change this to a number that would be capable for your computer to handle, as it is the number of training batches it will do at once. For reference, if you are not using cuda, i recommend a batch size of 4-8, and if you are using a nvidia gpu that is cuda capable (most rtx cards) you can change this number to something much larger (you'll have to play around with it to find the best number for your hardware)

Near the end of the config file, you should see the following parameters ```fine_tune_checkpoint```   ```label_map_path```   ```input_path``` and they may be repeated in different classes of the config file.
For ```fine_tune_checkpoint```
For ```label_map_path``` in training
For ```input_path``` in training
For ```g``` in eval (validate)
For ```g``` in eval (validate)

At the end of the file you should also find the parameter ```fine_tune_checkpoint_type```. Make sure this is set to ```"detection"``` and not ```"classification"``` or something else

### Actually beginning the training

To begin the actual training of the model using the mobilnet base model (as an example), you must run the command 

```bash
python model_main_tf2.py --pipeline_config_path=mobilnet640/pipeline.config --model_dir=training --alsologtostderr
```

while this is running, to view the training status status and model performance, open a seperate instance of anaconda prompt, activate the working environment, and run the command ```tensorboard --logdir=training``` in a seperate window/instance of anaconda prompt

together this should look like
```bash
conda activate tf2
tensorboard --logdir=training
```


## Quick Start (Legacy)

### Colabs

<!-- mdlint off(URL_BAD_G3DOC_PATH) -->

*   Training -
    [Fine-tune a pre-trained detector in eager mode on custom data](colab_tutorials/eager_few_shot_od_training_tf2_colab.ipynb)

*   Inference -
    [Run inference with models from the zoo](colab_tutorials/inference_tf2_colab.ipynb)

*   Few Shot Learning for Mobile Inference -
    [Fine-tune a pre-trained detector for use with TensorFlow Lite](colab_tutorials/eager_few_shot_od_training_tflite.ipynb)

<!-- mdlint on -->

## Training and Evaluation

To train and evaluate your models either locally or on Google Cloud see
[instructions](object_detection/g3doc/tf2_training_and_evaluation.md).

## Model Zoo

We provide a large collection of models that are trained on COCO 2017 in the
[Model Zoo](object_detection/g3doc/tf2_detection_zoo.md).

## Guides

*   <a href='object_detection/g3doc/configuring_jobs.md'>
      Configuring an object detection pipeline</a><br>
*   <a href='object_detection/g3doc/preparing_inputs.md'>Preparing inputs</a><br>
*   <a href='object_detection/g3doc/defining_your_own_model.md'>
      Defining your own model architecture</a><br>
*   <a href='object_detection/g3doc/using_your_own_dataset.md'>
      Bringing in your own dataset</a><br>
*   <a href='object_detection/g3doc/evaluation_protocols.md'>
      Supported object detection evaluation protocols</a><br>
*   <a href='object_detection/g3doc/tpu_compatibility.md'>
      TPU compatible detection pipelines</a><br>
*   <a href='object_detection/g3doc/tf2_training_and_evaluation.md'>
      Training and evaluation guide (CPU, GPU, or TPU)</a><br>
