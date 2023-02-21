# Yolov7-tflite-conversion

## Features

* Conversion of onnx yolov7 model (object detector) to tflite format
* Making predictions by using onnx model
* Make predictions by using tflite model

## Getting started

### Requirements
* Python = `3.9`
* Packages included in `requirements.txt` file
* Anaconda for an easy installation (not necessary)

### Clone repos:
* Clone this repo
* Clone original repo of yolov7:
```
$ git clone https://github.com/WongKinYiu/yolov7.git
```

### Environment

1) Create and activate a virtual environment:
```sh
$ conda create -n yolo7 python=3.9 anaconda;
$ conda activate yolo7
```

2) Install packages into the virtual environment:
```sh
$ cd yolov7-tflite-conversion;
$ pip install -r requirements.txt
```

3) Additional installation may be necessary - for onnx export:
```
$ pip --quiet install onnx onnxruntime onnxsim
$ pip install onnx-tf
```

4) You may also install full torch and tensorflow packages from official websites.

Note: in this repo an enveronment.yaml file is also included, which was produced by conda export manager. Above steps could be also reproduced by using only the following command:

```
$ conda env create --file environment.yaml
```

## Use

1) Go to original repo of yolov7 and download pytorch model, e.g.:

```
$ wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt
```

2) Convert pytorch yolo model to onnx format (from original repo):

```
$ python export.py --weights yolov7.pt --grid --end2end --simplify --topk-all 100 --iou-thres 0.65 --conf-thres 0.35 --img-size 640 640 --max-wh 640
```

3.) Move produced onnx yolov7 model to `models` directory of this repo.


4.) Convert the onnx model to tflite by using `converter/convert_to_tflite.py`

5.) Add arbitrary video or image to data directory (which you want to make predictions on).

6.) Make inference with the onnx model by using `onnx_predict.py`
or with tflite model by using `tflite_predict.py`


