# from keras.models import load_model
# from custom_module import CustomLayer
#
# model = load_model('path_to_your_model.h5', custom_objects={'CustomLayer': CustomLayer})
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the model
model = load_model('.\model_output.h5')

# Print the model summary
model.summary()