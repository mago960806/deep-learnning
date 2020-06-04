import numpy as np
from keras.datasets import mnist
from keras.models import load_model
from keras.engine.sequential import Sequential

_, (test_images, test_lables) = mnist.load_data()


test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype("float32") / 255

model: Sequential = load_model("models/model.hdf5")
results = np.argmax(model.predict(test_images), axis=1)
