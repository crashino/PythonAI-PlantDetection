import tensorflow as tf
from tensorflow.keras import models, layers
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageFilter

X = np.load("dataset.npy")
y = np.load("labels.npy")

X = X / 255.0

print(X.shape)
print(X[0])

model = models.Sequential()
model.add(layers.Conv2D(100, (3, 3), activation='relu', input_shape=(100, 100, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(200, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(200, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(200, activation='relu'))
model.add(layers.Dense(2))

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(X, y, epochs=40)


size = (100, 100)

def toarray(path):
    image = Image.open(path)
    image = image.convert('RGB')
    imageres = image.resize((size), Image.ANTIALIAS)
    blurred = imageres.filter(ImageFilter.GaussianBlur(radius=1.4))
    array = np.array(blurred)
    print(array.shape)
    #array = array.flatten()
    #array = array.reshape(1000000,)
    return array

test = np.empty((1, 100, 100, 3))

array = toarray('spider-plant-care-guide-header.jpg')

test[0] = array / 255.0

print(model.predict_classes(test))




