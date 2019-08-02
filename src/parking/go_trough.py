import skimage
from os.path import join
from os import listdir
from PIL import Image, ImageDraw
from pickle import dump

from colour import Color

from parking.folder_parsing_jpgs_and_labels import jpgs_and_labels
from parking.parkinglot_calc import read_config, get_masks_for_parking_lots, calc_occupancy_level
from parking.predict import filter_out_classes, predict_image_masks, merge_masks
from parking.mrcnn.visualize import display_instances
from parking.config import PATH_TO_DATASET, CLASS_NAMES


if __name__ == '__main__':

    PATH_TO_DATASET_IMAGES = '/Users/michael/Desktop/presentation/pieces/'
    file_names = [fn for fn in listdir(PATH_TO_DATASET_IMAGES) if fn.endswith('.jpg')]

    pklot_config = read_config()

    green = Color("green")
    gradient_colors = list(green.range_to(Color("red"), 100))

    for file_name in file_names:
        image_path = join(PATH_TO_DATASET_IMAGES, file_name)
        image = skimage.io.imread(image_path)
        res = filter_out_classes(predict_image_masks([image]))[0]
        #display_instances(image, res['rois'], res['masks'], res['class_ids'], CLASS_NAMES)
        single_mask = merge_masks(res['masks'])

        img = Image.open(image_path)

        for data in get_masks_for_parking_lots(pklot_config):
            confidence = calc_occupancy_level(data['mask'], single_mask)
            color = tuple([int(c * 256) for c in gradient_colors[int(confidence * 100)].get_rgb()] + [125, ])
            ImageDraw.Draw(img, 'RGBA').polygon(data['vertices'], fill=color)
            # img = Image.fromarray(data['mask'])

        # img.show()
        img.save(join(PATH_TO_DATASET_IMAGES, 'processed', file_name), 'JPEG')