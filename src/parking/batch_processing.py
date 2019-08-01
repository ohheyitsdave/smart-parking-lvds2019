import skimage
from os import path
from PIL import Image
from pickle import dump

from parking.folder_parsing_jpgs_and_labels import jpgs_and_labels
from parking.predict import filter_out_classes, predict_image_masks, merge_masks
from parking.mrcnn.visualize import display_instances
from parking.config import PATH_TO_DATASET, CLASS_NAMES


if __name__ == '__main__':

    PATH_TO_DATASET_IMAGES = path.join(PATH_TO_DATASET, 'images')
    length = int(len(jpgs_and_labels(PATH_TO_DATASET_IMAGES)[0]))

    images = []
    image_position = 0
    batches = 1

    while image_position <= length:
        for i in range(image_position, image_position + 21):
            image_path = jpgs_and_labels(PATH_TO_DATASET)[0][i]
            images.append(skimage.io.imread(image_path))

        res = filter_out_classes(predict_image_masks([images]))

        for i in range(len(images)):
            display_instances(images[i], res['rois'], res['masks'], res['class_ids'], CLASS_NAMES)
            single_mask = merge_masks(res['masks'])
            dump(single_mask, open('temp/single_mask' + batches + i + '.pkl', 'wb'))
            img = Image.fromarray(single_mask)
            img.save('/Users/michael/Downloads/single_mask' + batches + i + '.jpg', 'JPEG')

        image_position += 20
        batches += 1
