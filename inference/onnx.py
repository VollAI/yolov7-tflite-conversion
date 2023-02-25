import cv2
import matplotlib.pyplot as plt
import onnxruntime as ort

from configs import configs
from inference.helpers import load_image, pretty_image


class YoloOnnxModel:
    def __init__(self, model_name, use_cuda=False):
        model_path = f"{configs.BASE_DIR}/models/{model_name}.onnx"
        self.inname, self.outname, self.session = self._get_onnx_translator(model_path, use_cuda)

    def _get_onnx_translator(self, model_path, use_cuda):
        # Loading the ONNX inference session.
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_cuda else ["CPUExecutionProvider"]
        )
        session = ort.InferenceSession(model_path, providers=providers)
        # Getting onnx graph input and output names.
        outname = [i.name for i in session.get_outputs()]
        inname = [i.name for i in session.get_inputs()]
        return inname, outname, session

    def predict(self, image):
        inp = {self.inname[0]: image}
        prediction = self.session.run(self.outname, inp)[0]
        return prediction


if __name__ == "__main__":
    MODEL_NAME = "yolov7-tiny"
    IMAGE_PATH = "./data/whole_court_above_indoor.jpg"

    img = cv2.imread(IMAGE_PATH)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img, im, ratio, dwdh = load_image(img)

    model = YoloOnnxModel(MODEL_NAME)
    outputs = model.predict(im)

    names = configs.NAMES_COCO
    image = pretty_image(img, outputs, names, ratio, dwdh)
    plt.figure()
    plt.imshow(image)
    plt.show()
