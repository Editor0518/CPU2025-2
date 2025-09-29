import tensorflow as tf

print("TensorFlow 버전:", tf.__version__)
gpus = tf.config.list_physical_devices('GPU')


print("GPU 인식됨:", gpus)
