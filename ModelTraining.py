import numpy as np
from PIL import ImageFilter
from PIL import Image
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import joblib


tmp = np.genfromtxt("dataset.csv", delimiter=',')
i = (tmp.shape[0])

combined = [[[[]]]]
j = 0
while j < i:
    #print(tmp[j])
    tmparray = np.asarray([tmp[j]])
    tmparray = tmparray.reshape(125, 125, 3)
    #print(tmparray)

    if j == 0:
        combined = np.vstack(([tmparray[j]]))
        j += 1
    else:
        combined = np.vstack((combined, tmparray[j]))
        j += 1

print(combined)
print(combined[0])
X = np.genfromtxt("dataset.csv", delimiter=',')
y = np.genfromtxt("Labels.csv", delimiter=',')


mlp = MLPClassifier()

print("Training")
mlp.fit(X, y)
print("trained")


#mlp = joblib.load("model.pkl")

size = (125, 125)

def toarray(path):
    image = Image.open(path)
    image = image.convert('RGB')
    imageres = image.resize((size), Image.ANTIALIAS)
    blurred = imageres.filter(ImageFilter.GaussianBlur(radius=1.4))
    array = np.array(blurred)
    print(array.shape)
    array = array.flatten()
    #array = array.reshape(1000000,)
    return array

tester = toarray("parlor-palm-getty-0820-2000.jpg")

print(float(mlp.predict(tester.reshape(1, -1))))

joblib.dump(mlp, "model.pkl")