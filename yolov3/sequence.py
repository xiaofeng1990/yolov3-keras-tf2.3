from tensorflow.keras.utils import Sequence
from tensorflow.keras import backend as K
import math
import os
from PIL import Image
import numpy as np
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
import tensorflow as tf

class SequenceData(Sequence):
    def __init__(self, path, input_shape, anchors, num_classes, batch_size=128, max_boxes=20, shuffle=True):
        self.datasets = []
        with open(path, "r") as f:
            self.datasets = f.readlines()
        self.input_shape = input_shape
        self.batch_size = batch_size

        self.anchors = anchors
        self.shuffle = shuffle
        self.num_classes = num_classes
        self.max_boxes = max_boxes
        self.num_anchors = len(self.anchors)
        self.indexes = np.arange(len(self.datasets))

    def __len__(self):
        # 计算每一个epoch的迭代次数
        num_images = len(self.datasets)
        return math.ceil(num_images / float(self.batch_size)) -1

    def __getitem__(self, item):
        # 生成batch_size个索引
        batch_indexs = self.indexes[item * self.batch_size:(item + 1) * self.batch_size]
        # 根据索引获取datas集合中的数据
        batch = [self.datasets[k] for k in batch_indexs]
        # 生成数据
        X, y = self.data_generation(batch)
        return X, y

    def get_epochs(self):
        return self.__len__()

    def on_epoch_end(self):
        # 在每一次epoch结束是否需要进行一次随机，重新随机一下index
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def data_generation(self, batch):
        images = []
        boxes = []
        for data in batch:
            # 获取图片数据和box
            image, box = self.get_random_data(data,random=False)
            images.append(image)
            boxes.append(box)
        images = np.array(images)
        # batch, 20, 5
        boxes = np.array(boxes)
        y_true = self.preprocess_true_boxes(boxes)
        return [images, *y_true], np.zeros(self.batch_size)


    def get_random_data(self, data, random=True, jitter=.3, hue=.1, sat=1.5, val=1.5, proc_img=True):
        """
        random preprocessing for real-time data augmentation，并对数据增强
        :param data: 一行标注数据
        :param random: 随机
        :param jitter: 图片抖动参数
        :param hue: 色彩变换
        :param sat:
        :param val:
        :param proc_img:
        :return:
        """
        line = data.split()
        image = Image.open(line[0])
        # 图像实际尺寸
        iw, ih = image.size
        # 模型输入尺寸
        h, w = self.input_shape
        # 获得所有box坐标
        box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])

        if not random:
            # resize image，计算缩放比例，按照最长计算比例，保证图片不变性
            scale = min(w / iw, h / ih)
            # 缩放后的图片
            nw = int(iw * scale)
            nh = int(ih * scale)
            # 计算填充的长度和宽度
            dx = (w - nw) // 2
            dy = (h - nh) // 2
            image_data = 0
            if proc_img:
                # 将图像缩放
                image = image.resize((nw, nh), Image.BICUBIC)
                # 灰底图片
                new_image = Image.new('RGB', (w, h), (128, 128, 128))
                # 将缩放后的图片贴到灰底图片上
                new_image.paste(image, (dx, dy))
                # 图片归一化
                image_data = np.array(new_image) / 255.0
            # correct boxes
            box_data = np.zeros((self.max_boxes, 5))
            if len(box) > 0:
                np.random.shuffle(box)
                if len(box) > self.max_boxes:
                    box = box[:self.max_boxes]
                # 将原图的坐标进行缩放然后加上上半部的灰度坐标偏移
                box[:, [0, 2]] = box[:, [0, 2]] * scale + dx
                box[:, [1, 3]] = box[:, [1, 3]] * scale + dy
                box_data[:len(box)] = box

            return image_data, box_data

        # resize image
        new_ar = w / h * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
        scale = self.rand(.25, 2)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)

        image = image.resize((nw, nh), Image.BICUBIC)
        dx = int(self.rand(0, w - nw))
        dy = int(self.rand(0, h - nh))
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image = new_image
        # flip image or not,随机反转图片
        flip = self.rand() < .5
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            # tf.image.flip_left_right(image)
            # 显示

        # distort image
        hue = self.rand(-hue, hue)
        sat = self.rand(1, sat) if self.rand() < .5 else 1 / self.rand(1, sat)
        val = self.rand(1, val) if self.rand() < .5 else 1 / self.rand(1, val)
        x = rgb_to_hsv(np.array(image) / 255.)
        x[..., 0] += hue
        x[..., 0][x[..., 0] > 1] -= 1
        x[..., 0][x[..., 0] < 0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x > 1] = 1
        x[x < 0] = 0
        image_data = hsv_to_rgb(x)  # numpy array, 0 to 1
        # correct boxes
        box_data = np.zeros((self.max_boxes, 5))
        if len(box) > 0:
            np.random.shuffle(box)
            box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
            box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
            if flip:
                box[:, [0, 2]] = w - box[:, [2, 0]]
            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > w] = w
            box[:, 3][box[:, 3] > h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box
            if len(box) > self.max_boxes:
                box = box[:self.max_boxes]
            box_data[:len(box)] = box

        return image_data, box_data

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def preprocess_true_boxes(self, true_boxes):
        """
        Preprocess true boxes to training input format
        :param true_boxes: array, shape=(m, T, 5) m表示批次大小，T表示一张图像里允许最多目标，5表示(x1, y1, x2. y2, class) -->(32, 20, 5)
        :return: y_true: list of array, shape like yolo_outputs, xywh are reletive value
        """
        assert (true_boxes[..., 4] <self.num_classes).all(), 'class id must be less than num_classes'
        num_layers = self.num_anchors // 3  # default setting
        anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [1, 2, 3]]
        # (32, 20, 5)
        true_boxes = np.array(true_boxes, dtype="float32")
        # (416,416)
        input_shape = np.array(self.input_shape, dtype="int32")
        # 计算中心点坐标 # (32, 20, 2)
        boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
        boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]
        # 计算中心点坐标关于原尺寸的比例
        true_boxes[..., 0:2] = boxes_xy / input_shape[::-1]
        # 计算w和h相对于原图的比例
        true_boxes[..., 2:4] = boxes_wh / input_shape[::-1]
        # m 表示批次
        m = true_boxes.shape[0]
        # 计算不同特征图上的网格尺寸，网格越多，表示检测小目标[[13*13],[26*26], [52, 52]]
        grid_shapes = [input_shape // {0: 32, 1: 16, 2: 8}[l] for l in range(num_layers)]
        # 生成网格 y_true 里面有3个网格
        y_true = [np.zeros((m, grid_shapes[l][0], grid_shapes[l][1], len(anchor_mask[l]), 5 + self.num_classes), dtype="float32")
                  for l in range(num_layers)]
        # Expand dim to apply broadcasting.
        anchors = np.expand_dims(self.anchors, 0)  # (1,9,2)
        # 求的anchor 的宽高的一半作为右下角坐标，取反为左上角坐标， anchor中点为坐标原点
        anchor_maxes = anchors / 2.
        anchor_mins = -anchor_maxes
        # 宽度大于0的才算有效  (32,20)
        valid_mask = boxes_wh[..., 0] > 0

        for b in range(m):
            # Discard zero rows. (2, 2) 得到不是0的方框
            wh = boxes_wh[b, valid_mask[b]]
            if len(wh) == 0: continue
            # Expand dim to apply broadcasting. 立方体左边的一个竖面
            wh = np.expand_dims(wh, -2)
            # 按照方框中心为原点，计算左上角坐标，和右下角坐标
            box_maxes = wh / 2.
            box_mins = -box_maxes
            # 获得交集左上角坐标 (2,9,2),让每一个方框的长宽可所有anchor的长宽比较，得到两者最大的
            intersect_mins = np.maximum(box_mins, anchor_mins)
            # 交集右下角坐标
            intersect_maxes = np.minimum(box_maxes, anchor_maxes)
            # 计算交集的w h
            intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
            # 计算交集面积 (2, 9)
            intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
            # 计算方框面积 (2, 9)
            box_area = wh[..., 0] * wh[..., 1]
            # 计算anchor box 面积 (1, 9)
            anchor_area = anchors[..., 0] * anchors[..., 1]
            # 计算交并比 (2, 9)
            iou = intersect_area / (box_area + anchor_area - intersect_area)
            # Find best anchor for each true box 得到 iou的行表示方框索引，列表示对应anchor的iou的值，axis=-1，表示得到行中iou的最大值
            # 索引可以表示anchor的index
            best_anchor = np.argmax(iou, axis=-1)
            # t表示方框索引， n表示anchor index
            for t, n in enumerate(best_anchor):
                for l in range(num_layers):  # 特征图索引
                    if n in anchor_mask[l]:  # 判断anchor 是否在该特征图
                        # 计算方格坐标
                        i = np.floor(true_boxes[b, t, 0] * grid_shapes[l][1]).astype('int32')
                        j = np.floor(true_boxes[b, t, 1] * grid_shapes[l][0]).astype('int32')
                        k = anchor_mask[l].index(n)  # 获得anchor mask中的索引 可以当前特征图上的anchor 的 index
                        c = true_boxes[b, t, 4].astype('int32')
                        y_true[l][b, j, i, k, 0:4] = true_boxes[b, t, 0:4]
                        y_true[l][b, j, i, k, 4] = 1
                        y_true[l][b, j, i, k, 5 + c] = 1

        return y_true