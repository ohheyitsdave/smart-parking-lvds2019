import cv2

from parking.config import PATH_TO_COCO_WEIGHTS, PATH_TO_MODEL_STRUCTURE

net = cv2.dnn.readNetFromTensorflow(PATH_TO_COCO_WEIGHTS, PATH_TO_MODEL_STRUCTURE)