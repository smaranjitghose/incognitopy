# python blur_face.py --image examples/adrian.jpg --face face_detector --method pixelated

# import the necessary packages
import numpy as np
import argparse
import cv2
import os


def anonymize_face_pixelate(image, blocks=3):
	# divide the input image into NxN blocks
	(h, w) = image.shape[:2]
	xSteps = np.linspace(0, w, blocks + 1, dtype="int")
	ySteps = np.linspace(0, h, blocks + 1, dtype="int")
	# loop over the blocks in both the x and y direction
	for i in range(1, len(ySteps)):
		for j in range(1, len(xSteps)):
			# compute the starting and ending (x, y)-coordinates
			# for the current block
			startX = xSteps[j - 1]
			startY = ySteps[i - 1]
			endX = xSteps[j]
			endY = ySteps[i]
			# extract the ROI using NumPy array slicing, compute the
			roi = image[startY:endY, startX:endX]
			# mean of the ROI, and then draw a rectangle with the
			(B, G, R) = [int(x) for x in cv2.mean(roi)[:3]]
			# mean RGB values over the ROI in the original image
			cv2.rectangle(image, (startX, startY), (endX, endY),(B, G, R), -1)

	# return the pixelated blurred image
	return image


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,help="path to input image")
ap.add_argument("-b", "--blocks", type=int, default=20,help="# of blocks for the pixelated blurring method")
ap.add_argument("-c", "--confidence", type=float, default=0.5,help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load serialized face detector model 
print("[INFO] Fetching Face Detector...")
net = cv2.dnn.readNet('deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')


image = cv2.imread(args["image"])
# Get Spatial Dimensions
(h, w) = image.shape[:2]

# construct a blob from the image
blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),(104.0, 177.0, 123.0))

# pass the blob through the network and obtain the face detections
print("[INFO] Looking for those lovely faces...")
net.setInput(blob)
detections = net.forward()

# loop over the detections
for i in range(0, detections.shape[2]):
	# extract the confidence (i.e., probability) associated with the detection
	confidence = detections[0, 0, i, 2]
	# filter out weak detections by ensuring the confidence is greater than the minimum confidence
	if confidence > args["confidence"]:
		# compute the (x, y)-coordinates of the bounding box for the object
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")
		# extract the Region of Interest for the face
		face = image[startY:endY, startX:endX]
		# Blur the face in Region of Interest
		face = anonymize_face_pixelate(face,blocks=args["blocks"])
		# Replace the original face with the blurred face in the ROI
		image[startY:endY, startX:endX] = face

# Display the pixelated image
cv2.imshow("IncognitoPY", image)
# Input keypress
k = cv2.waitKey(0)
if k == ord('q'):
    # Save the image in the desired path
    cv2.imwrite('blurred.png', image)
    #close all the opened windows
    cv2.destroyAllWindows()

