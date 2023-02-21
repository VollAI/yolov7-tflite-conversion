import cv2
import matplotlib.pyplot as plt
import tensorflow as tf

from configs import configs
from inference.helpers import load_image, pretty_image


class BaseTFLiteModel:
    def __init__(self, model_path):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

    def predict(self, input_image):
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()
        self.interpreter.set_tensor(input_details[0]["index"], input_image.numpy())
        self.interpreter.invoke()
        prediction = self.interpreter.get_tensor(output_details[0]["index"])
        return prediction


class YoloTFLiteModel:
    def __init__(self, model_name):
        self.model = BaseTFLiteModel(f"{configs.BASE_DIR}/models/{model_name}.tflite")

    def predict(self, image):
        input_image = tf.cast(image, dtype=tf.float32)
        prediction = self.model.predict(input_image)
        return prediction


if __name__ == "__main__":
    MODEL_NAME = "yolov7-tiny"
    IMAGE_PATH = "./data/whole_court_above_indoor.jpg"

    img = cv2.imread(IMAGE_PATH)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img, im, ratio, dwdh = load_image(img)

    model = YoloTFLiteModel(MODEL_NAME)
    outputs = model.predict(im)

    names = configs.NAMES_COCO
    image = pretty_image(img, outputs, names, ratio, dwdh)
    plt.figure()
    plt.imshow(image)
    plt.show()
