import tensorflow as tf
print("TensorFlow 버전:", tf.__version__)
tf.debugging.set_log_device_placement(True)

print("사용 가능한 디바이스:", tf.config.list_physical_devices('GPU'))
