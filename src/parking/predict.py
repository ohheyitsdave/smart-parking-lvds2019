# -*- coding: utf-8 -*-
from functools import reduce

import parking.coco as coco
import skimage.io

import parking.mrcnn.model as modellib
from parking.mrcnn.visualize import display_instances
from parking.config import COCO_MODEL_PATH, CLASS_NAMES, ACCEPTED_CLASSES_INDEXES


class InferenceConfig(coco.CocoConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MAX_INSTANCES = 1000
    DETECTION_MIN_CONFIDENCE = 0.35
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)
    IMAGE_MIN_DIM = 1024
    IMAGE_MAX_DIM = 2048


config = InferenceConfig()
config.display()


def _merge_masks(masks):
    """
    :param masks: list of true/false masks shape (image height, image width, image num)
    :type masks: np.array
    :return: single mask merged
    :rtype: np.array
    """
    return reduce(lambda x, y: x | y, [masks[:, :, i] for i in range(masks.shape[2])])


def _filter_out_classes(predictions, accepted_classes=ACCEPTED_CLASSES_INDEXES):
    filtered = []
    for result in predictions:
        new_result = {}
        indexes = [i for i, class_id in enumerate(result['class_ids']) if class_id in accepted_classes]
        new_result['masks'] = result['masks'][:, :, indexes]
        new_result['class_ids'] = result['class_ids'][indexes]
        new_result['scores'] = result['scores'][indexes]
        new_result['rois'] = result['rois'][indexes]
        filtered.append(new_result)
    return filtered


def predict_image_masks(images):
    """
    :type images: list[imageio.core.util.Array]

    :return: list of mappings with rois masks and predicted class indexes for each image
    :rtype: list[dict]
    """

    model = modellib.MaskRCNN(mode="inference", model_dir='./', config=config)
    model.load_weights(COCO_MODEL_PATH, by_name=True)
    return model.detect(images, verbose=False)


if __name__ == '__main__':
    image_path = '/Users/michael/work/lvivds/parkingslot/images/2018-07-16 07:45:39.024.jpg'
    image = skimage.io.imread(image_path)

    res = _filter_out_classes(predict_image_masks([image]))[0]

    display_instances(image, res['rois'], res['masks'], res['class_ids'], CLASS_NAMES)

    from PIL import Image, ImageDraw

    single_mask = _merge_masks(res['masks'])
    img = Image.fromarray(single_mask)
    img.save('/Users/michael/Downloads/single_mask2.jpg', 'JPEG')


# single_mask = reduce(lambda x, y: x | y, [result['masks'][:, :, i] for i in indexes])

# print(single_mask)


# from json import load
# pklot_config_file = '/Users/michael/work/lvivds/smart-parking-lvds2019/notebooks/pklot_config.json'
# pklot_config = load(open(pklot_config_file))
# d = pklot_config[0]['annotations'][0]
# tupVerts = list(zip([float(px) for px in d['xn'].split(';')], [float(py) for py in d['yn'].split(';')]))
#
# x, y = np.meshgrid(np.arange(single_mask.shape[0]), np.arange(single_mask.shape[1])) # make a canvas with coordinates
# x, y = x.flatten(), y.flatten()
# points = np.vstack((x, y)).T
#
# p = Path(tupVerts)
# grid = p.contains_points(points)
# mask = grid.reshape(single_mask.shape[0], single_mask.shape[1]) # now you have a mask with poi
#
# print(mask)
# display_instances(image, results[0]['rois'], results[0]['masks'], results[0]['class_ids'], CLASS_NAMES)
# print([class_names[i] for i in results[0]['class_ids']])
# img.save('/Users/michael/Downloads/processed2.jpg', 'JPEG')