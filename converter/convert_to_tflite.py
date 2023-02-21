import os
import tempfile
from pathlib import Path

import onnx
import tensorflow as tf
from onnx_tf.backend import prepare

from configs import configs

# see other possible options for optimization:
# https://www.tensorflow.org/lite/performance/post_training_quantization

# set input name of onnx model, without suffix!
MODEL_NAME = "yolov7"


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)
    return dirname


if __name__ == "__main__":
    base_dir = configs.BASE_DIR
    ensure_dir(f"{base_dir}/models/")

    with tempfile.TemporaryDirectory(dir=base_dir) as dp:
        # os.system(f"onnx-tf convert -i {base_dir}/models/{MODEL_NAME}.onnx -o {dp}")
        # OR:
        onnx_model = onnx.load(f"{base_dir}/models/{MODEL_NAME}.onnx")  # load onnx model
        tf_rep = prepare(onnx_model)  # prepare tf representation
        tf_rep.export_graph(dp)  # export the model

        converter = tf.lite.TFLiteConverter.from_saved_model(dp)
        # use optimization: see link above
        # converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()

        with open(f"{base_dir}/models/{MODEL_NAME}.tflite", "wb") as f:
            f.write(tflite_model)
