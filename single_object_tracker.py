import cv2 as cv
import numpy as np
import argparse
import time 

# PARSING ARGUEMENTS
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-l", "--labels", required=True,
	help="path to ImageNet labels (i.e., syn-sets)")
args = vars(ap.parse_args())

# LOADING THE INPUT IMAGE FROM DISK

image = cv.imread(args["image"])

# LOAD THE CLASS LABELS FROM DISK

rows = open(args["labels"]).read().strip().split("\n")
classes = [r[r.find(" ") + 1:].split(",")[0] for r in rows]

# CONVERTING IT TO DNN usable format
blob = cv.dnn.blobFromImage(image, 1, (224, 224), (104, 117, 123))

# LOADING THE MODEL
net = cv.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# SET THE BLOB AS THE NETWORK INPUT
net.setInput(blob)
preds = net.forward()

# SORTING THE TOP 5 CLASSIFICATION AND PRINTING IT

idxs = np.argsort(preds[0])[::-1][:5]
for (i, idx) in enumerate(idxs):
    	# draw the top prediction on the input image
	if i == 0:
		text = "Label: {}, {:.2f}%".format(classes[idx],
			preds[0][idx] * 100)
		cv.putText(image, text, (5, 25),  cv.FONT_HERSHEY_SIMPLEX,
			0.7, (0, 0, 255), 2)
	print("{}. label: {}, probability: {:.5}".format(i + 1, classes[idx], preds[0][idx]))


cv.imshow("IMAGE",image)
cv.waitKey(0)