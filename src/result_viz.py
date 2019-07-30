from xml.etree import ElementTree
from os.path import join

from PIL import Image, ImageDraw

path_to_pkl_data = '/Users/michael/Downloads/PKLot'

description_file_path = 'PKLot/PUCPR/Sunny/2012-09-11/2012-09-11_15_16_58.xml'
lot_image_file_path = 'PKLot/PUCPR/Sunny/2012-09-11/2012-09-11_15_16_58.jpg'
tree = ElementTree.parse(join(path_to_pkl_data, description_file_path))

root = tree.getroot()

source_img = Image.open(join(path_to_pkl_data, lot_image_file_path)).convert("RGB")
draw = ImageDraw.Draw(source_img)

for space in root:
    print(space.attrib['id'], space.get('occupied'))
    rotated_rect, contour = space
    center, side, angle = rotated_rect
    points = [p.attrib for p in contour]
    print(center.attrib, side.attrib, angle.attrib)
    print(points)
    if space.get('occupied') is not None:
        draw.polygon([(int(p['x']), int(p['y'])) for p in points], outline='red' if int(space.get('occupied')) else 'green')

source_img.save('/Users/michael/Downloads/test.jpg', 'JPEG')
print(source_img)
