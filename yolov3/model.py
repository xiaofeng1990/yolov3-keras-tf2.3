"""YOLO_v3 Model Defined in Keras."""
from functools import wraps

import numpy as np
import tensorflow as tf

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D, Add, ZeroPadding2D, UpSampling2D, Concatenate, MaxPooling2D
from tensorflow.keras.layers import LeakyReLU, BatchNormalization
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2

from yolov3.utils import compose


# 装饰函数，保证被装饰的函数的固有属性不被装饰器函数改变, Conv2D中的一些参数不被DarknetConv2D改变
@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    """Wrapper to set Darknet parameters for Convolution2D."""
    # kernal 采用l2正则
    darknet_conv_kwargs = {"kernel_regularizer": l2(5e-4)}
    # 根据strides 决定padding
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides') == (2, 2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)


# *args：位置参数，个数不定, **kwargs：关键字参数 key=value
def DarknetConv2D_BN_Leaky(*args, **kwargs):
    """
    Darknet Convolution2D followed by BatchNormalization and LeakyReLU.
    固件一个卷积快，conv2d+BN+Leaky
    """
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1))


def resblock_body(x, num_filters, num_blocks):
    """
    A series of resblocks starting with a downsampling Convolution2D
    一个下采样快, 然后完成 shortcut
    :param x: 输入层
    :param num_filters: filters 数量
    :param num_blocks: blocks 数量
    :return:
    """
    # Darknet uses left and top padding instead of 'same' mode
    x = ZeroPadding2D(((1, 0), (1, 0)))(x)
    # 卷积快， filter=32，kernel=3*3, padding = valid, strides = 2 下采样
    x = DarknetConv2D_BN_Leaky(num_filters, (3, 3), strides=(2, 2))(x)
    for i in range(num_blocks):
        y = compose(
            DarknetConv2D_BN_Leaky(num_filters // 2, (1, 1)),
            DarknetConv2D_BN_Leaky(num_filters, (3, 3)))(x)
        # shortcut
        x = Add()([x, y])
    return x


def darknet_body(x):
    """Darknent body having 52 Convolution2D layers"""
    # 第一个卷积层次 filter=32，kernel=3*3, padding = same, strides = 1
    x = DarknetConv2D_BN_Leaky(32, (3, 3))(x)
    x = resblock_body(x, 64, 1)
    x = resblock_body(x, 128, 2)
    x = resblock_body(x, 256, 8)
    x = resblock_body(x, 512, 8)
    x = resblock_body(x, 1024, 4)
    return x


def make_last_layers(x, num_filters, out_filters):
    """
    6 Conv2D_BN_Leaky layers followed by a Conv2D_linear layer
    :param x: 输入层
    :param num_filters:
    :param out_filters:
    :return:
    """
    x = compose(
        DarknetConv2D_BN_Leaky(num_filters, (1, 1)),
        DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3)),
        DarknetConv2D_BN_Leaky(num_filters, (1, 1)),
        DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3)),
        DarknetConv2D_BN_Leaky(num_filters, (1, 1)))(x)
    y = compose(
        DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3)),
        DarknetConv2D(out_filters, (1, 1)))(x)
    return x, y


def yolo_body(inputs, num_anchors, num_classes):
    """
    Create YOLO_V3 model CNN body in Keras.
    :param inputs: 输入层
    :param num_anchors: 一个 特征图的 anchors 数量，这里一个特征图有3个anchors
    :param num_classes: 分类数量
    :return: model 一个输入，三个输出
    """
    darknet = Model(inputs, darknet_body(inputs))
    x, y1 = make_last_layers(darknet.output, 512, num_anchors * (num_classes + 5))
    # rout -4
    x = compose(
        DarknetConv2D_BN_Leaky(256, (1, 1)),
        UpSampling2D(2))(x)
    # rout -1 61
    x = Concatenate()([x, darknet.layers[152].output])
    x, y2 = make_last_layers(x, 256, num_anchors * (num_classes + 5))

    # rout -4
    x = compose(
        DarknetConv2D_BN_Leaky(128, (1, 1)),
        UpSampling2D(2))(x)
    # -1, 36
    x = Concatenate()([x, darknet.layers[92].output])
    x, y3 = make_last_layers(x, 128, num_anchors * (num_classes + 5))

    return Model(inputs, [y1, y2, y3])


def yolo_head(feats, anchors, num_classes, input_shape, calc_loss=False):
    """
    Convert final layer features to bounding box parameters
    :param feats: 模型输出的三个特征图中的一个(b, grid, grid, anchor, 25) (x, y, w, h, c, p20)
    :param anchors: 当前特征图上的三个anchor
    :param num_classes: 类别数量
    :param input_shape: 输入尺寸
    :param calc_loss:
    :return:
    """
    num_anchors = len(anchors)  # 3
    # Reshape to batch, height, width, num_anchors, box_params. (1, 1, 3, 2)
    anchors_tensor = K.reshape(K.constant(anchors), [1, 1, num_anchors, 2])
    # height, width 获取网格形状
    grid_shape = K.shape(feats)[1:3]
    # K.arange(0, stop=grid_shape[0]) -->[ 0  1  2  3  4  5  6  7  8  9 10 11 12] (13,)
    # K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]) --> (13, 1, 1, 1)
    # (grid, grid, 1, 1)
    grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
                    [1, grid_shape[1], 1, 1])
    # K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1] -->(1, 13, 1, 1)
    # (grid, grid, 1, 1)
    grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
                    [grid_shape[0], 1, 1, 1])
    # (grid, grid, 1, 2)
    grid = K.concatenate([grid_x, grid_y])
    grid = K.cast(grid, K.dtype(feats))
    # (1, grid, grid, 3, 25)
    feats = K.reshape(
        feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])
    # Adjust preditions to each spatial grid point and anchor size.
    # (b, grid, grid, anchor, 2)
    box_xy = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[..., ::-1], K.dtype(feats))
    # (b, grid, grid, anchor, 2)
    box_wh = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[..., ::-1], K.dtype(feats))
    # (b, grid, grid, anchor, 1)
    box_confidence = K.sigmoid(feats[..., 4:5])
    # (b, grid, grid, anchor, 20)
    box_class_probs = K.sigmoid(feats[..., 5:])

    if calc_loss == True:
        return grid, feats, box_xy, box_wh

    return box_xy, box_wh, box_confidence, box_class_probs


def box_iou(b1, b2):
    """
    Return iou tensor
    :param b1: tensor, shape=(grid, grid, anchor, 4), xywh
    :param b2: tensor, shape=(none, none), xywh
    :return: iou: tensor, shape=(b, grid, grid, anchor, j)
    """
    # Expand dim to apply broadcasting， 扩展 anchor (grid, grid, anchor,1, 4)
    b1 = K.expand_dims(b1, -2)
    b1_xy = b1[..., :2]  # (grid, grid, anchor,1, 4)
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh / 2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half

    # Expand dim to apply broadcasting.扩展 batch (1, none, none)
    b2 = K.expand_dims(b2, 0)
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh / 2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half
    # (none, none, 3, none 2)
    intersect_mins = K.maximum(b1_mins, b2_mins)
    intersect_maxes = K.minimum(b1_maxes, b2_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]  # (none, none, 3, none)
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]  # (none, none, 3, none, 1)
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]  # (1, none)
    iou = intersect_area / (b1_area + b2_area - intersect_area)  # (none, none, 3, none)

    return iou


def yolo_loss(args, anchors, num_classes, ignore_thresh=.5, print_loss=False):
    """

    :param args:
        yolo_outputs: list of tensor, the output of yolo_body or tiny_yolo_body
        y_true: list of array, the output of preprocess_true_boxes
    :param anchors: array, shape=(N, 2), wh
    :param num_classes:
    :param ignore_thresh: float, the iou threshold whether to ignore object confidence loss
    :param print_loss:
    :return: loss: tensor, shape=(1,)
    """
    # 获得特征图数量 3
    num_layers = len(anchors) // 3
    # 获得模型输出 3个 (none, none, none, 75)
    yolo_outputs = args[:num_layers]
    # 获得真实值 3个(b, grid, grid, anchor, 25) (x, y, w, h, c, p20)
    y_true = args[num_layers:]
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [1, 2, 3]]
    input_shape = K.cast(K.shape(yolo_outputs[0])[1:3] * 32, K.dtype(y_true[0]))  # 416*416
    grid_shapes = [K.cast(K.shape(yolo_outputs[l])[1:3], K.dtype(y_true[0])) for l in
                   range(num_layers)]  # 13*13，26*26， 52*52
    loss = 0

    # batch size tensor
    m = K.shape(yolo_outputs[0])[0]
    mf = K.cast(m, K.dtype(yolo_outputs[0]))

    for l in range(num_layers):
        # 获取置信度 (batch, grid, grid, anchor, c) -->(32, 13, 13, 3, 1)
        object_mask = y_true[l][..., 4:5]
        # 获取概率值 (batch, grid, grid, anchor, p20) -->(32, 13, 13, 3, 20)
        true_class_probs = y_true[l][..., 5:]
        # grid (grid, grid, 1, 2)
        grid, raw_pred, pred_xy, pred_wh = yolo_head(yolo_outputs[l],
                                                     anchors[anchor_mask[l]], num_classes, input_shape, calc_loss=True)

        # (b, grid, grid, anchor, 4) x,y,w,h
        pred_box = K.concatenate([pred_xy, pred_wh])

        # Darknet raw box to calculate loss.
        # 可能是计算在的是偏移量 (b, grid, grid, anchor, 2)
        raw_true_xy = y_true[l][..., :2] * grid_shapes[l][::-1] - grid
        raw_true_wh = K.log(y_true[l][..., 2:4] / anchors[anchor_mask[l]] * input_shape[::-1])
        raw_true_wh = K.switch(object_mask, raw_true_wh, K.zeros_like(raw_true_wh))  # avoid log(0)=-inf
        # 不知道啥意思 (b, grid, grid, anchor, 1)
        box_loss_scale = 2 - y_true[l][..., 2:3] * y_true[l][..., 3:4]

        # Find ignore mask, iterate over each of batch.
        # 分配一个动态数组
        ignore_mask = tf.TensorArray(K.dtype(y_true[0]), size=1, dynamic_size=True)
        object_mask_bool = K.cast(object_mask, "bool")

        def loop_body(b, ignore_mask):
            # (b, grid, grid, anchor, 25)   (32, grid, grid, anchor, 1)，删除掉没有box的anchor
            # (grid, grid, x, 4) （none, none）
            true_box = tf.boolean_mask(y_true[l][b, ..., 0:4], object_mask_bool[b, ..., 0])
            iou = box_iou(pred_box[b], true_box)
            best_iou = K.max(iou, axis=-1)  # (none, none, 3)
            ignore_mask = ignore_mask.write(b, K.cast(best_iou < ignore_thresh, K.dtype(true_box)))
            return b + 1, ignore_mask

        # b 是当前循环索引，m是一个批次总数，
        _, ignore_mask = tf.while_loop(lambda b, *args: b < m, loop_body, [0, ignore_mask])
        ignore_mask = ignore_mask.stack()
        # (none, none, none, 3, 1)
        ignore_mask = K.expand_dims(ignore_mask, -1)

        # K.binary_crossentropy is helpful to avoid exp overflow.
        xy_loss = object_mask * box_loss_scale * K.binary_crossentropy(raw_true_xy, raw_pred[..., 0:2],
                                                                       from_logits=True)
        wh_loss = object_mask * box_loss_scale * 0.5 * K.square(raw_true_wh - raw_pred[..., 2:4])
        confidence_loss = object_mask * K.binary_crossentropy(object_mask, raw_pred[..., 4:5], from_logits=True) + \
                          (1 - object_mask) * K.binary_crossentropy(object_mask, raw_pred[..., 4:5],
                                                                    from_logits=True) * ignore_mask
        class_loss = object_mask * K.binary_crossentropy(true_class_probs, raw_pred[..., 5:], from_logits=True)

        xy_loss = K.sum(xy_loss) / mf
        wh_loss = K.sum(wh_loss) / mf
        confidence_loss = K.sum(confidence_loss) / mf
        class_loss = K.sum(class_loss) / mf
        loss += xy_loss + wh_loss + confidence_loss + class_loss
        if print_loss:
            loss = tf.print(loss, [loss, xy_loss, wh_loss, confidence_loss, class_loss, K.sum(ignore_mask)],
                            message='loss: ')
        return loss


def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape):
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = K.cast(input_shape, K.dtype(box_yx))
    image_shape = K.cast(image_shape, K.dtype(box_yx))
    new_shape = K.round(image_shape * K.min(input_shape / image_shape))
    offset = (input_shape - new_shape) / 2. / input_shape
    scale = input_shape / new_shape
    box_yx = (box_yx - offset) * scale
    box_hw *= scale
    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes = K.concatenate([
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]  # x_max
    ])
    # Scale boxes back to original image shape.
    boxes *= K.concatenate([image_shape, image_shape])
    return boxes


def yolo_boxes_and_scores(feats, anchors, num_classes, input_shape, image_shape):
    """
    Process Conv layer output
    :param feats:
    :param anchors:
    :param num_classes:
    :param input_shape:
    :param image_shape:
    :return:
    """
    # 将feats 转换为 tensor
    feats = tf.convert_to_tensor(feats, dtype=tf.float32)
    # 解析出feats中的信息
    box_xy, box_wh, box_confidence, box_class_probs = yolo_head(feats,
                                                                anchors, num_classes, input_shape)

    # 计算真实方框 (1, grid, grid, 3, 4)
    boxes = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape)
    # (grid*grid, 4)
    boxes = K.reshape(boxes, [-1, 4])
    box_scores = box_confidence * box_class_probs # (grid, grid, 3, 1)* (grid, grid, 3, 20)
    box_scores = K.reshape(box_scores, [-1, num_classes]) # (grid*grid,  20)
    return boxes, box_scores


def yolo_eval(yolo_outputs,
              anchors,
              num_classes,
              image_shape,
              max_boxes=20,
              score_threshold=.6,
              iou_threshold=.5):
    """
    计算模型输出
    :param yolo_outputs: 模型输出
    :param anchors: anchors list
    :param num_classes: 类别数目
    :param image_shape: 图形形状 (行， 列)
    :param max_boxes: 单张图片中做多方框
    :param score_threshold:
    :param iou_threshold:
    :return:
    """
    num_layers = len(yolo_outputs)
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [1, 2, 3]]  # default setting
    input_shape = K.shape(yolo_outputs[0])[1:3] * 32
    boxes = []
    box_scores = []
    for l in range(num_layers):
        # 解析boxes 和scores
        _boxes, _box_scores = yolo_boxes_and_scores(yolo_outputs[l], anchors[anchor_mask[l]], num_classes, input_shape,
                                                    image_shape)
        boxes.append(_boxes)
        box_scores.append(_box_scores)
    # (10647, 4)
    boxes = K.concatenate(boxes, axis=0)
    # (10647, 20)
    box_scores = K.concatenate(box_scores, axis=0)
    mask = box_scores >= score_threshold  # (10647, 20)
    max_boxes_tensor = K.constant(max_boxes, dtype='int32')

    boxes_ = []
    scores_ = []
    classes_ = []
    for c in range(num_classes):
        # 根据mask过滤预测到c类的所有单元格 tensor(,4)，如果没有，就为空的tensor(0,4)
        class_boxes = tf.boolean_mask(boxes, mask[:, c])
        # tensor(,)
        class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])
        nms_index = tf.image.non_max_suppression(
            class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=iou_threshold)
        print(nms_index)
        # 获取索引为nms_index的方框
        class_boxes = K.gather(class_boxes, nms_index)
        class_box_scores = K.gather(class_box_scores, nms_index)
        classes = K.ones_like(class_box_scores, 'int32') * c
        boxes_.append(class_boxes)
        scores_.append(class_box_scores)
        classes_.append(classes)
    # 将tensor连接，空的tensor会删除掉， boxes_, scores_, classes_ 的位置是一一对应
    boxes_ = K.concatenate(boxes_, axis=0)
    scores_ = K.concatenate(scores_, axis=0)
    classes_ = K.concatenate(classes_, axis=0)

    return boxes_, scores_, classes_