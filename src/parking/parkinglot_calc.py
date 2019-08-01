from os.path import join
from json import load

import numpy as np

from PIL import Image, ImageDraw

from parking.config import IMAGE_SHAPE, PATH_TO_DATASET


def read_config():
    pklot_config_file = join(PATH_TO_DATASET, 'pklot_config.json')
    return load(open(pklot_config_file))


def get_masks_for_parking_lots(pklot_config):
    for d in pklot_config[0]['annotations']:
        img = Image.new('L', (IMAGE_SHAPE[0], IMAGE_SHAPE[1]), 0)
        vert = list(zip([float(px) for px in d['xn'].split(';')], [float(py) for py in d['yn'].split(';')]))
        ImageDraw.Draw(img).polygon(vert, fill='white')
        yield {'id': d['id'], 'mask': np.array(img).astype(bool)}


def calc_occupancy_level(slot_mask, cars_mask):
    slot_area = slot_mask.sum()
    ocupied_place_area = (slot_mask & cars_mask).sum()
    return ocupied_place_area / slot_area


if __name__ == '__main__':
    from itertools import islice

    pklot_config = read_config()

    from pickle import load
    single_mask = load(open('temp/single_mask.pkl', 'rb'))
    for data in islice(get_masks_for_parking_lots(pklot_config), 10):
        print(data['id'], calc_occupancy_level(data['mask'], single_mask))
        #img = Image.fromarray(data['mask'])
        #img.show()
