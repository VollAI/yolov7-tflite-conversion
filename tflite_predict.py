import cv2
import numpy as np

from configs import configs
from inference.helpers import generate_frames, load_image, pretty_image
from inference.tflite import YoloTFLiteModel

MODEL_NAME = "yolov7-tiny"
VIDEO_PATH = "./data/half_court_sunny1.mp4"
DRAW_CLASSES = ["sports ball"]


if __name__ == "__main__":
    frame_generator = generate_frames(VIDEO_PATH)
    model = YoloTFLiteModel(MODEL_NAME)

    for img in frame_generator:
        img, im, ratio, dwdh = load_image(img)

        outputs = model.predict(im)

        names = configs.NAMES_COCO
        image = pretty_image(img, outputs, names, ratio, dwdh, draw_classes=DRAW_CLASSES)

        cv2.imshow("image", image)
        cv2.waitKey(1)
