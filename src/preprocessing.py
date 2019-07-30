from xml.etree import ElementTree
from os.path import join

path_to_pkl_data = '/Users/michael/Downloads/PKLot'

tree = ElementTree.parse(join(path_to_pkl_data, 'PKLot/PUCPR/Sunny/2012-09-11/2012-09-11_15_16_58.xml'))

root = tree.getroot()

for space in root:
    print(space.attrib['id'], space.get('occupied'))
    rotated_rect, contour = space
    center, side, angle = rotated_rect
    points = [p.attrib for p in contour]
    print(center.attrib, side.attrib, angle.attrib)
    print(points)


source_img = Image.open(file_name).convert("RGB")