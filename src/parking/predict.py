import os
import random
import pickle

from itertools import islice

import cv2
import numpy as np

from parking.config import PATH_TO_COCO_WEIGHTS, PATH_TO_MODEL_STRUCTURE, COLORS, CLASSES

CONFIDENCE = 0
MASK_THRESHOLD = 0.3

net = cv2.dnn.readNetFromTensorflow(PATH_TO_COCO_WEIGHTS, PATH_TO_MODEL_STRUCTURE)


def get_images():
    root_path = '/Users/michael/Downloads/smart_parking_unique/images/'
    for root, dirs, files in os.walk(root_path):
        for filename in files:
            yield os.path.join(root, filename)

    # yield '/Users/michael/Downloads/test_croped6.jpg'
    # yield '/Users/michael/Downloads/test_croped7.jpg'
    #yield '/Users/michael/Desktop/test_croped.jpg'

suffix = 'regular'

for img_number, path_to_image in enumerate(islice(get_images(), 4, 5)):

    image = cv2.imread(path_to_image)
    (H, W) = image.shape[:2]

    # resized = cv2.resize(image, (W*2, H*2))
    # image=resized
    # (H, W) = image.shape[:2]

    # construct a blob from the input image and then perform a forward
    # pass of the Mask R-CNN, giving us (1) the bounding box  coordinates
    # of the objects in the image along with (2) the pixel-wise segmentation
    # for each specific object
    blob = cv2.dnn.blobFromImage(image, swapRB=False, crop=False)
    net.setInput(blob)
    (boxes, masks) = net.forward(["detection_out_final", "detection_masks"])
    #pickle.dump({'boxes': boxes, 'masks': masks}, open('/Users/michael/Downloads/masks.data', 'wb'))
    # clone our original image so we can draw on it
    clone = image.copy()
    print(path_to_image)
    print(boxes.shape[2])
    for i in range(0, boxes.shape[2]):
        #  extract the class ID of the detection along with the confidence
        #  (i.e., probability) associated with the prediction
        classID = int(boxes[0, 0, i, 1])
        if classID != 2:
            continue

        confidence = boxes[0, 0, i, 2]

        # filter out weak predictions by ensuring the detected probability
        # is greater than the minimum probability
        if confidence > CONFIDENCE:
            # scale the bounding box coordinates back relative to the
            # size of the image and then compute the width and the height
            # of the bounding box
            box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H])
            (startX, startY, endX, endY) = box.astype("int")
            boxW = endX - startX
            boxH = endY - startY

            # extract the pixel-wise segmentation for the object, resize
            # the mask such that it's the same dimensions of the bounding
            # box, and then finally threshold to create a *binary* mask
            mask = masks[i, classID]
            mask = cv2.resize(mask, (boxW, boxH),
                              interpolation=cv2.INTER_LINEAR)
            mask = (mask > MASK_THRESHOLD)

            # extract the ROI of the image
            roi = clone[startY:endY, startX:endX]

            # convert the mask from a boolean to an integer mask with
            # to values: 0 or 255, then apply the mask
            visMask = (mask * 255).astype("uint8")
            instance = cv2.bitwise_and(roi, roi, mask=visMask)

            # show the extracted ROI, the mask, along with the
            # segmented instance
            # cv2.imshow("ROI", roi)
            # cv2.imshow("Mask", visMask)
            # cv2.imshow("Segmented", instance)

            # now, extract *only* the masked region of the ROI by passing
            # in the boolean mask array as our slice condition
            roi = roi[mask]

            # randomly select a color that will be used to visualize this
            # particular instance segmentation then create a transparent
            # overlay by blending the randomly selected color with the ROI
            color = random.choice(COLORS)
            blended = ((0.4 * color) + (0.6 * roi)).astype("uint8")

            # store the blended ROI in the original image
            clone[startY:endY, startX:endX][mask] = blended

            # draw the bounding box of the instance on the image
            color = [int(c) for c in color]
            cv2.rectangle(clone, (startX, startY), (endX, endY), color, 2)

            # draw the predicted label and associated probability of the
            # instance segmentation on the image
            text = "{}: {:.4f}".format(CLASSES[classID], confidence)
            cv2.putText(clone, text, (startX, startY - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imwrite(f'/Users/michael/Downloads/segmented{img_number}_{suffix}_{str(boxes.shape[2])}.jpg', clone)