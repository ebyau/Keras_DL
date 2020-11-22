import matplotlib

matplotlib.use("Agg")

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from pyimagesearch.smallervggnet import SmallerVGGNet
import matplotlib.pyplot as plt
from imutils import paths
import tensorflow as tf
import numpy as np
import argparse
import random
import pickle
import cv2
import os

# construct argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to input dataset(image directory)")
ap.add_argument("-m", "--model", required=True,
                help="path to output model")
ap.add_argument("-l", "--labelbin", required= True,
                help="path to output label binarizer")
ap.add_argument("-p", "--plot", type=str, default='plot.png',
                help="path to output accuracy/loss plot")
args = vars(ap.parse_args())

EPOCHS = 30
INIT_LR = 1e-3  # learning rate
BS = 32  # Batch Size
IMAGE_DIMS = (96, 96, 3)

tf.compat.v1.disable_eager_execution()

# get image paths and randomly shuffle them
print("[INFO]  Loading images .....")
imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.seed(42)
random.shuffle(imagePaths)

# initialize data and labels
data = []
labels = []

#loop over the input images
for imagePath in imagePaths:
    # load the image, preprocess it, and store it in the data list
    image = cv2.imread(imagePath)
    image = cv2.resize(image,(IMAGE_DIMS[1], IMAGE_DIMS[0]))
    image = img_to_array(image)
    data.append(image)
    # extract set of class labels from image path
    # and update the labels list
    l = label = imagePath.split(os.path.sep)[-2].split("_")
    labels.append(l)
# scale the raw pixels to range [0,1]
data = np.array(data,dtype="float") / 255.0
labels = np.array(labels)
print("[INFO]  data matrix: {} images ({:.2f}MB".format(
        len(imagePaths), data.nbytes / (1024*1000.0)
    ))

# binarize the labels using scikit learn multi label binarizer
print("[INFO] class labels: ")
mlb = MultiLabelBinarizer()
labels = mlb.fit_transform(labels)

# looop over class labels and show them
for (i,label) in enumerate(mlb.classes_):
    print("{} {}".format(i+1,label))
# split data into train and test data
(trainX,testX,trainY,testY) = train_test_split(data,labels,
                                test_size=0.2,random_state=42)
# image generator for data augementation
aug = ImageDataGenerator(rotation_range=25,width_shift_range=0.1,
                             height_shift_range=0.1,shear_range=0.2,
                             zoom_range=0.2,
                             horizontal_flip=True,fill_mode="nearest")
# initialize model using sigmoid fnx
print("[INFO] compiling model .....")
model = SmallerVGGNet.build(
    width=IMAGE_DIMS[1], height=IMAGE_DIMS[0],
    depth=IMAGE_DIMS[2],classes=len(mlb.classes_),finalAct="sigmoid"
)

# initialize optimizer and compile model
opt = Adam(lr=INIT_LR,decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
              metrics=['accuracy'])

# Training the networkk
print("[INFO] training network .....")
H = model.fit(
    x= aug.flow(trainX,trainY,batch_size=BS),
    validation_data=(testX,testY),
    steps_per_epoch=len(trainX) // BS,
    epochs= EPOCHS, verbose =1)

# saving model to disk
print("[INFO] serializing network ......")
model.save(args['model'], save_format="h5")

# save multi-label binarizer to disk
print("[INFO] serializing label binarizer .....")
f = open(args['labelbin'], "wb")
f.write(pickle.dumps(mlb))
f.close()

#plot training loss vs accuracy
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")
plt.savefig(args["plot"])


















