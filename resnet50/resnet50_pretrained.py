import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import resnet50

model = resnet50.ResNet50(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)

file = "African_Elephant.jpg"

model_config = model.get_config()
width, height = model.inputs[0].shape.as_list()[1:3]

image = tf.keras.preprocessing.image.load_img( file, target_size=( width, height ) )
x = tf.keras.preprocessing.image.img_to_array( image )
x = np.expand_dims( x, axis = 0 )

x = resnet50.preprocess_input(x)
y = resnet50.decode_predictions( model.predict( x ), top = 1 )

print("Found: {}".format(y[0][0][1]))
