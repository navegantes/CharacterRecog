import os
import struct
from tkinter import Tk
from tkinter.filedialog import askdirectory
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import cv2

"""
Loosely inspired by http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
which is GPL licensed.
"""

def read(dataset="training", path="."):
    """
    Python function for importing the MNIST data set.  It returns an iterator
    of 2-tuples with the first element being the label and the second element
    being a numpy.uint8 2D array of pixel data for the given image.
    """

    if dataset == "training":
        fname_img = os.path.join(path, 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dataset == "testing":
        fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
    else:
        raise ValueError

    # Load everything in some numpy arrays
    with open(fname_lbl, 'rb') as flbl:
        _, _ = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)

    with open(fname_img, 'rb') as fimg:
        _, _, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)

    get_img = lambda idx: (lbl[idx], img[idx])

    # Create an iterator which returns each image in turn
    for i in range(len(lbl)):
        yield get_img(i)

def show(image):
    """
    Render a given numpy.uint8 2D array of pixel data.
    """
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    imgplot = ax.imshow(image, cmap=mpl.cm.Greys)
    imgplot.set_interpolation('nearest')
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')
    plt.show()
    plt.close()

def main():

    root = Tk()
    root.withdraw()

    datadir = askdirectory(parent=root, title="Enter with the directory...").__str__()

    root.destroy()

    imTrainset = list(read(dataset="training", path=datadir))

    img = imTrainset[0][1]
    show(img)

    kern = np.ones((2, 2), np.uint8)

    _, bimg = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    show(bimg)

    morph = cv2.morphologyEx(bimg, cv2.MORPH_CLOSE, kern, iterations=1)
    show(morph)

if __name__ == "__main__":
    main()
