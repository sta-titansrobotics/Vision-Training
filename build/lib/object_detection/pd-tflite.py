import tensorflow as tf


converter = tf.lite.TFLiteConverter.from_saved_model("inference_graph/saved_model")
converter.target_spec.supported_ops = [
tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
]
tflite_model = converter.convert()
open("inference_graph/saved_model/converted_model.tflite", "wb").write(tflite_model)