import tensorflow as tf
# Load the .pb file
converter = tf.lite.TFLiteConverter.from_saved_model(r"C:\Users\Geps08\.vscode\code files\ex_machina2\models\research\object_detection\inference_graph\saved_model")
# Convert the model
tflite_model = converter.convert()
# Save the .tflite file
with open(r"C:\Users\Geps08\.vscode\code files\ex_machina2\models\research\object_detection\inference_graph\saved_model\converted_model.tflite", "wb") as f:
   f.write(tflite_model)