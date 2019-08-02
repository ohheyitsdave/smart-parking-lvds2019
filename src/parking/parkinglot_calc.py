from os.path import join
from json import load

import numpy as np
import skimage

from PIL import Image, ImageDraw

from parking.config import IMAGE_SHAPE, PATH_TO_DATASET
from parking.predict import filter_out_classes, predict_image_masks, merge_masks


def read_config():
    pklot_config_file = join(PATH_TO_DATASET, 'pklot_config.json')
    return load(open(pklot_config_file))


def get_masks_for_parking_lots(pklot_config):
    for d in pklot_config[0]['annotations']:
        img = Image.new('L', (IMAGE_SHAPE[0], IMAGE_SHAPE[1]), 0)
        vert = list(zip([float(px) for px in d['xn'].split(';')], [float(py) for py in d['yn'].split(';')]))
        ImageDraw.Draw(img).polygon(vert, fill='white')
        yield {'id': d['id'], 'mask': np.array(img).astype(bool), 'vertices': vert}


def calc_occupancy_level(slot_mask, cars_mask):
    slot_area = slot_mask.sum()
    ocupied_place_area = (slot_mask & cars_mask).sum()
    return ocupied_place_area / slot_area


if __name__ == '__main__':
    from PIL import Image, ImageDraw
    image_path = '/Users/michael/work/lvivds/parkingslot/images/2018-07-16 07:45:39.024.jpg'

    img = Image.open(image_path)

    pklot_config = read_config()

    from colour import Color

    green = Color("green")
    gradient_colors = list(green.range_to(Color("red"), 100))

    res = filter_out_classes(predict_image_masks([skimage.io.imread(image_path)]))[0]
    single_mask = merge_masks(res['masks'])

    for data in get_masks_for_parking_lots(pklot_config):
        # Image.fromarray(data['mask']).save('/Users/michael/Desktop/presentation/slots_masks' + str(data['id'])+'.jpg', 'JPEG')
        confidence = calc_occupancy_level(data['mask'], single_mask)
        print(data['id'], confidence)
        color = tuple([int(c * 256) for c in gradient_colors[int(confidence * 100)].get_rgb()] + [125, ])
        ImageDraw.Draw(img, 'RGBA').polygon(data['vertices'], fill=color)
        #img = Image.fromarray(data['mask'])

    #img.show()
    img.save('/Users/michael/Desktop/presentation/predicted.jpg', 'JPEG')
