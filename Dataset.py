from PIL import Image
from PIL import ImageFilter
import numpy as np
import os
import matplotlib.pyplot as plt

size = (100, 100)  # a variable to change the size of photo resize


def toarray(path):  # defines a function that turns the photo (at location given) to a 3D array
    image = Image.open(path)  # opens photo
    image = image.convert('RGB')  # converts to rgb, removes alpha layer from jpgs
    imageres = image.resize((size), Image.ANTIALIAS)  # resizes photo to size given in size variable
    blurred = imageres.filter(ImageFilter.GaussianBlur(radius=1.4))  # Blurs photo so model is based more on colours
    array = np.array(blurred)  # makes an array from the blurred, resized value.
    #  print(array.shape)
    # array = array.flatten()
    # array = array.reshape(1000000,)
    return array


photocount = 0
for name in os.listdir("photos"):
    photocount += 1

print(photocount)
i = 0
combined = np.empty((photocount, 100, 100, 3))

yvals = []

for name in os.listdir("photos"):  # a loop that goes through each photo in the folder
    array = toarray("photos\\" + str(name))  # turn the photo into an array by using file name
    combined[i] = array
    i += 1

    list(name)  # convert the name of the a list
    y = int(name[0])  # Take the first character of the name, which is the label
    yvals.append(y)  # Add the label value to the yvals array.

print(combined.shape)
np.save("dataset.npy", combined)
np.save("labels.npy", yvals)

#  np.savetxt("dataset.csv", combined, fmt="%ld", delimiter=',')
#  np.savetxt("Labels.csv", yvals, fmt="%ld", delimiter=',')
