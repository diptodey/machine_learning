from keras import layers
from keras import models
import pickle




mnist_data = pickle.load( open( "mnist_data.p", "rb" ) )
model = models.Sequential()


model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))


model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))


model.compile(optimizer='rmsprop',loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(mnist_data['train_images'], mnist_data['train_labels'], epochs=5, batch_size=64)

test_loss, test_acc = model.evaluate(mnist_data['test_images'], mnist_data['test_labels'])
print(test_loss)
print(test_acc)