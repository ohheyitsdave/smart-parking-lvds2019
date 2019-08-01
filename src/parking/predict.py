# -*- coding: utf-8 -*-
from functools import reduce

import parking.coco as coco
import numpy as np
import skimage.io

import parking.mrcnn.model as modellib
from matplotlib.path import Path
from parking.mrcnn.visualize import display_instances
from parking.config import COCO_MODEL_PATH, CLASS_NAMES, ACCEPTED_CLASSES_INDEXES


class InferenceConfig(coco.CocoConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MAX_INSTANCES = 1000
    DETECTION_MIN_CONFIDENCE = 0.5
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)
   # IMAGE_RESIZE_MODE = 'pad64'
    IMAGE_MIN_DIM = 4096
    IMAGE_MAX_DIM = 4096


config = InferenceConfig()
config.display()

image_path = '/Users/michael/work/lvivds/parkingslot/images/2018-07-16 07:45:39.024.jpg'
image = skimage.io.imread(image_path)


def _merge_masks(masks):
    """
    :param masks: list of true/false masks shape (image height, image width, image num)
    :type masks: np.array
    :return:
    :rtype:
    """



def predict_image_masks(images, classes=ACCEPTED_CLASSES_INDEXES):
    """
    :type images: list[imageio.core.util.Array]

    :return: list of mappings with rois masks and predicted class indexes for each image
    :rtype: list[dict]
    """

    model = modellib.MaskRCNN(mode="inference", model_dir='./', config=config)
    model.load_weights(COCO_MODEL_PATH, by_name=True)
    return model.detect([image], verbose=False)


result = predict_image_masks([image])[0]


display_instances(image, result['rois'], result['masks'], result['class_ids'], CLASS_NAMES)

accepted_class = np.array([3, 6, 8, 9, 29, 68])
indexes = [i for i, class_id in enumerate(result['class_ids']) if class_id in ACCEPTED_CLASSES_INDEXES]

single_mask = reduce(lambda x, y: x | y, [result['masks'][:, :, i] for i in indexes])

print(single_mask)


from json import load
pklot_config_file = '/Users/michael/work/lvivds/smart-parking-lvds2019/notebooks/pklot_config.json'
pklot_config = load(open(pklot_config_file))
d = pklot_config[0]['annotations'][0]
tupVerts = list(zip([float(px) for px in d['xn'].split(';')], [float(py) for py in d['yn'].split(';')]))

x, y = np.meshgrid(np.arange(single_mask.shape[0]), np.arange(single_mask.shape[1])) # make a canvas with coordinates
x, y = x.flatten(), y.flatten()
points = np.vstack((x, y)).T

p = Path(tupVerts)
grid = p.contains_points(points)
mask = grid.reshape(single_mask.shape[0], single_mask.shape[1]) # now you have a mask with poi

print(mask)
# display_instances(image, results[0]['rois'], results[0]['masks'], results[0]['class_ids'], CLASS_NAMES)
# print([class_names[i] for i in results[0]['class_ids']])
# img.save('/Users/michael/Downloads/processed2.jpg', 'JPEG')