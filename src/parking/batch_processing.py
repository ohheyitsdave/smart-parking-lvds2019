import skimage

from src.parking.folder_parsing_jpgs_and_labels import jpgs_and_labels
from src.parking.predict import filter_out_classes, predict_image_masks


if __name__ == '__main__':
    PATH_TO_DATASET = 'D:/Personal/DS Studies/Lviv DS School 2019/Project/smart-parking-lvds2019/dataset/images'  #fix this later
    length = int(len(jpgs_and_labels(PATH_TO_DATASET)[0]) / 100) #fix this later, now decreased for testing
    for i in range(length):
        image_path = jpgs_and_labels(PATH_TO_DATASET)[0][i]
        image = skimage.io.imread(image_path)

        res = filter_out_classes(predict_image_masks([image]))[0]

        display_instances(image, res['rois'], res['masks'], res['class_ids'], CLASS_NAMES)

        single_mask = merge_masks(res['masks'])

        dump(single_mask, open('temp/single_mask' + i + '.pkl', 'wb'))

        img = Image.fromarray(single_mask)
        img.save('/Users/polina/Downloads/single_mask' + i + '.jpg', 'JPEG')

