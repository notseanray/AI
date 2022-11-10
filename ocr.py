from pyimagesearch.models import ResNet
from pyimagesearch.az_dataset import load_mnist_dataset
from pyimagesearch.az_dataset import load_az_dataset
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import build_montages
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import argparse
matplotlib.use("Agg")

ap = argparse.ArgumentParser(
    prog = 'Epic OCR',
    description = 'do the ocr',
    epilog = 'optical')

ap.add_argument("-a", "--az", required=True,
    help="path to A-Z dataset")
ap.add_argument("-m", "--model", type=str, required=True,
    help="path to output trained handwriting recognition model")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
    help="path to output training history file")
args = vars(ap.parse_args())

EPOCHS = 50
INIT_LR = 1e-1
BS = 128
(azData, azLabels) = load_az_dataset(args["az"])
(digitsData, digitsLabels) = load_mnist_dataset()

labels = np.hstack([azLabels, digitsLabels])
data = np.array(azData, dtype="float32")
data = np.expand_dims(data, axis=-1)
data /= 255.0

labels = le.fit_transform(labels)
counts = labels.sum(axis=0)
classWeight = {}
# classWeight[i] = classTotals.max() / classTotals[i]
(trainX, testX, trainY, testY) = train_test_split(
    data,
    labels,
    test_size=0.20,
    stratify=labels,
    random_state=42
    rotation_range=10,
    zoom_range=0.05,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.15,
    horizontal_flip=False,
    fill_mode="nearest"
)

opt = SGD(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model = ResNet.build(32, 32, 1, len(le.classes_), (3, 3, 3),
    (64, 64, 128, 256), reg=0.0005)
model.compile(loss="categorical_crossentropy", optimizer=opt,
    metrics=["accuracy"])

H = model.fit(
    aug.flow(trainX, trainY, batch_size=BS),
    validation_data=(testX, testY),
    steps_per_epoch=len(trainX) // BS,
    epochs=EPOCHS,
    class_weight=classWeight,
    verbose=1)
labelNames = azLabels
labelNames += "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
labelNames = [l for l in labelNames]
predictions = model.predict(testX, batch_size=BS)
print(classification_report(testY.argmax(axis=1),
    predictions.argmax(axis=1), target_names=labelNames))

model.save(args["model"], save_format="h5")
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.title("Training Loss and Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])
prediction = predictions.argmax(axis=1)
label = labelNames[prediction[0]]
image = (testX[i] * 255).astype("uint8")
color = (0, 0, 255)
image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_LINEAR)
cv2.putText(image, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
montage = build_montages(image, (96, 96), (7, 7))[0]
cv2.waitKey(0)

def load_az_dataset(datasetPath):
    data = []
    labels = []

    for row in open(datasetPath):
        row = row.split(",")
        label = int(row[0])
        image = np.array([int(x) for x in row[1:]], dtype="uint8")

        image = image.reshape((28, 28))

        data.append(image)
        labels.append(label)

        data = np.array(data, dtype="float32")
        labels = np.array(labels, dtype="int")
    return (data, labels)


def load_mnist_dataset():
    ((trainData, trainLabels), (testData, testLabels)) = mnist.load_data()
    data = np.vstack([trainData, testData])
    labels = np.hstack([trainLabels, testLabels])
    return (data, labels)

