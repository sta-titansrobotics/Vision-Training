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
#check installed version on conda env (if this outputs a version other than 3.20.3, you did something wrong and must redo the previous step or find further help)
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

## Quick Start (Legacy)

### Colabs

<!-- mdlint off(URL_BAD_G3DOC_PATH) -->

*   Training -
    [Fine-tune a pre-trained detector in eager mode on custom data](../colab_tutorials/eager_few_shot_od_training_tf2_colab.ipynb)

*   Inference -
    [Run inference with models from the zoo](../colab_tutorials/inference_tf2_colab.ipynb)

*   Few Shot Learning for Mobile Inference -
    [Fine-tune a pre-trained detector for use with TensorFlow Lite](../colab_tutorials/eager_few_shot_od_training_tflite.ipynb)

<!-- mdlint on -->

## Training and Evaluation

To train and evaluate your models either locally or on Google Cloud see
[instructions](tf2_training_and_evaluation.md).

## Model Zoo

We provide a large collection of models that are trained on COCO 2017 in the
[Model Zoo](tf2_detection_zoo.md).

## Guides

*   <a href='configuring_jobs.md'>
      Configuring an object detection pipeline</a><br>
*   <a href='preparing_inputs.md'>Preparing inputs</a><br>
*   <a href='defining_your_own_model.md'>
      Defining your own model architecture</a><br>
*   <a href='using_your_own_dataset.md'>
      Bringing in your own dataset</a><br>
*   <a href='evaluation_protocols.md'>
      Supported object detection evaluation protocols</a><br>
*   <a href='tpu_compatibility.md'>
      TPU compatible detection pipelines</a><br>
*   <a href='tf2_training_and_evaluation.md'>
      Training and evaluation guide (CPU, GPU, or TPU)</a><br>
