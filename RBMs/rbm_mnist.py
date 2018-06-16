from keras.datasets import mnist
import numpy as np

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

print(np.shape(train_images))
print(train_images[0])