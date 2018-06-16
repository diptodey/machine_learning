from keras.datasets import mnist
from keras.utils import to_categorical
import pickle
from keras.preprocessing import sequence

"""
The problem we’re trying to solve here is to classify grayscale images of handwritten
digits (28 × 28 pixels) into their 10 categories (0 through 9). We’ll use the MNIST
dataset, a classic in the machine-learning community, which has been around almost
as long as the field itself and has been intensively studied. It’s a set of 60,000 training
images, plus 10,000 test images, assembled by the National Institute of Standards and
Technology (the NIST in MNIST) in the 1980s
"""
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

#introducing a new label 10th as invalid

mnist_data = {}




"""
train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)


mnist_data['train_images'] = train_images
mnist_data['test_images'] = test_images
mnist_data['train_labels'] = train_labels
mnist_data['test_labels'] = test_labels


pickle.dump( mnist_data, open( "mnist_data.p", "wb" ) )
"""