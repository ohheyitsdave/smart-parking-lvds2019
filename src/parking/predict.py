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
    DETECTION_MAX_INSTANCES = 500
    DETECTION_MIN_CONFIDENCE = 0.5


config = InferenceConfig()
config.display()

image_path = '/Users/michael/work/lvivds/parkingslot/images/2018-07-16 07:45:39.024.jpg'
image = skimage.io.imread(image_path)

model = modellib.MaskRCNN(mode="inference", model_dir='./', config=config)
model.load_weights(COCO_MODEL_PATH, by_name=True)
result = model.detect([image], verbose=False)[0]

# print(results)

# from PIL import Image, ImageDraw
# img = Image.open(image_path)
# draw = ImageDraw.Draw(img)
# for i, class_id in enumerate(results[0]['class_ids']):
#     # if class_id !=3:
#     #     continue
#     box_coord = results[0]['rois'][i]
#     draw.rectangle([box_coord[1], box_coord[0], box_coord[3], box_coord[2]], outline='red', width=3)
#     draw.text([box_coord[1], box_coord[0]], class_names[class_id])

accepted_class = np.array([3, 6, 8, 9, 29, 68])
indexes = [i for i, class_id in enumerate(result['class_ids']) if class_id in accepted_class]

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