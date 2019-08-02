import skimage
from os import path
from PIL import Image
from pickle import dump

from parking.folder_parsing_jpgs_and_labels import jpgs_and_labels
from parking.predict import filter_out_classes, predict_image_masks, merge_masks
from parking.mrcnn.visualize import display_instances
from parking.config import PATH_TO_DATASET, CLASS_NAMES


if __name__ == '__main__':

    PATH_TO_DATASET_IMAGES = '/Users/michael/Desktop/presentation/pieces/'
    length = int(len(jpgs_and_labels(PATH_TO_DATASET_IMAGES)[0]))

    images = []
    image_position = 0
    k = 0

    while image_position <= length:
        for i in range(image_position, image_position + 3):
            image_path = jpgs_and_labels(PATH_TO_DATASET_IMAGES)[0][i]
            print(image_path)
            images.append(skimage.io.imread(image_path))
        print(len(images))
        res = filter_out_classes(predict_image_masks(images))

        for i in range(len(images)):
            display_instances(images[i], res['rois'], res['masks'], res['class_ids'], CLASS_NAMES)
            single_mask = merge_masks(res['masks'])
            img = Image.fromarray(single_mask)
            img.save(f'/Users/michael/Downloads/single_mask{k}.jpg', 'JPEG')
            k += 1

        image_position += 3
        break
