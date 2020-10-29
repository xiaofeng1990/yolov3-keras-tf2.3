import xml.etree.ElementTree as ET
from os import getcwd
import argparse

sets = [('2007', 'train'), ('2007', 'val'), ('2007', 'test')]

classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
           "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

parser = argparse.ArgumentParser(description="Pascal-VOC dta path.")
parser.add_argument('pascal_path', help='Path to Pascal-VOC data set.')


def convert_annotation(pascal_path, year, image_id, list_file):
    in_file = open('%s/VOCdevkit/VOC%s/Annotations/%s.xml' % (pascal_path, year, image_id))
    tree = ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text),
             int(xmlbox.find('ymax').text))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))


# wd = getcwd()
# python voc_annotation.py /home/ubuntu/work/dataset/Pascal-VOC
if __name__ == "__main__":
    # Pascal path
    args = parser.parse_args()
    pascal_path = args.pascal_path

    for year, image_set in sets:
        image_ids = open(
            '%s/VOCdevkit/VOC%s/ImageSets/Main/%s.txt' % (pascal_path, year, image_set)).read().strip().split()
        list_file = open('%s_%s.txt' % (year, image_set), 'w')
        for image_id in image_ids:
            list_file.write('%s/VOCdevkit/VOC%s/JPEGImages/%s.jpg' % (pascal_path, year, image_id))
            convert_annotation(pascal_path, year, image_id, list_file)
            list_file.write('\n')
        list_file.close()
