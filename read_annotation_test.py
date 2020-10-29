import numpy as np

from yolov3.utils import get_random_data
from yolov3.model import preprocess_true_boxes


def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes):
    """
    数据迭代器
    :param annotation_lines:
    :param batch_size:
    :param input_shape:
    :param anchors:
    :param num_classes:
    :return:
    """
    n = len(annotation_lines)
    i = 0
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            if i == 0:
                np.random.shuffle(annotation_lines)
            # box 20 * 5
            image, box = get_random_data(annotation_lines[i], input_shape, random=True)
            image_data.append(image)
            box_data.append(box)
            i = (i + 1) % n
        # (batch, w, h, c)-->(32, 416, 416, 3)
        image_data = np.array(image_data)
        box_data = np.array(box_data) # (32, 20, 5)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        yield [image_data, *y_true], np.zeros(batch_size)




def data_generator_wrapper(annotation_lines, batch_size, input_shape, anchors, num_classes):
    """
    数据生成器
    :param annotation_lines: 标注数据行
    :param batch_size: 一个批次数据量
    :param input_shape: 输入模型的图像大小
    :param anchors: 锚定框
    :param num_classes: 类别数量
    :return:
    """
    n = len(annotation_lines)
    if n == 0 or batch_size <= 0:
        return None
    return data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes)


def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


def _main():
    annotation_path = '2007_train.txt'
    log_dir = 'logs/000/'
    classes_path = 'model/voc_classes.txt'
    anchors_path = 'model/yolo_anchors.txt'
    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)
    input_shape = (416, 416)  # multiple of 32, hw

    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines) * val_split)
    num_train = len(lines) - num_val
    batch_size = 32
    generator = data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes)
    for [image_data, y_true], label in generator:
        print(image_data.shape)
        print(y_true)



if __name__ == "__main__":
    _main()
