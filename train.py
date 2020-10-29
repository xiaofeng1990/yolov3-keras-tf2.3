import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model
# 优化器
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from yolov3.model import yolo_body, yolo_loss
from yolov3.utils import get_classes, get_anchors
from yolov3.sequence import SequenceData

import os
import argparse
import sys
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def parse_arguments(argv):
    parser = argparse.ArgumentParser(description='train yolov3 model.')
    # 训练文件, 必须参数
    parser.add_argument('train_file', type=str, help='Path to train data file.')
    # 验证文件 必须参数
    parser.add_argument('val_file', type=str, help='Path to val data file.')
    # 预训练模型路径 必须参数
    parser.add_argument('model_file', type=str, help='pre-training model  weights file.')
    # voc classes 文件路径
    parser.add_argument("--classes_file", type=str, default="model/voc_classes.txt",
                        help="Describe the type of classification")
    # yolo anchors 文件路径
    parser.add_argument("--anchors_file", type=str, default="model/yolo_anchors.txt",
                        help="Describe anchor")
    # 输出日志路径
    parser.add_argument("--logs_path", type=str, default="logs/", help="log file path")
    # 训练 batch_size 8的倍数
    parser.add_argument("--batch_size", type=int, default=128, help="batch size.")
    # 训练 epochs
    parser.add_argument("--epochs", type=int, default=200, help="train epochs.")

    return parser.parse_args(argv)


def create_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
                 weights_path='model/yolo_weights.h5'):
    """
    create the training model

    :param input_shape: 输入图片尺寸
    :param anchors: anchor 列表
    :param num_classes: classes 数量
    :param load_pretrained: 是否加载预训练模型
    :param freeze_body: 是否冻结部分层次
    :param weights_path: 预训练模型路径
    :return:
    """
    K.clear_session()
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)
    # [<tf.Tensor 'input_2:0' shape=(None, 13, 13, 3, 25) dtype=float32>,
    # <tf.Tensor 'input_3:0' shape=(None, 26, 26, 3, 25) dtype=float32>,
    # <tf.Tensor 'input_4:0' shape=(None, 52, 52, 3, 25) dtype=float32>]
    y_true = [
        Input(shape=(h // {0: 32, 1: 16, 2: 8}[l], w // {0: 32, 1: 16, 2: 8}[l], num_anchors // 3, num_classes + 5))
        for
        l in range(3)]
    model_body = yolo_body(image_input, num_anchors // 3, num_classes)
    print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))
    # 加载域训练权重
    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze darknet53 body or freeze all but 3 output layers.
            num = (185, len(model_body.layers) - 3)[freeze_body - 1]
            for i in range(num):
                model_body.layers[i].trainable = False

            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    # todo 自定义loss
    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
                        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
        [*model_body.output, *y_true])

    model = Model([model_body.input, *y_true], model_loss)

    return model


def _main(args):
    # todo 训练数据
    annotation_train_file = args.train_file
    assert os.path.exists(annotation_train_file), "train file {} is not exists".format(annotation_train_file)
    annotation_val_file = args.val_file
    assert os.path.exists(annotation_val_file), "val file {} is not exists".format(annotation_val_file)
    model_file = args.model_file
    assert model_file.endswith('.h5'), '{} is not a .cfg file'.format(model_file)
    # 验证日志目录
    logs_path = args.logs_path
    if not os.path.exists(logs_path):
        os.mkdir(logs_path)

    # todo classes and anchors 数据
    classes_file = args.classes_file
    assert os.path.exists(classes_file), "classes file {} is not exists".format(classes_file)
    anchors_file = args.anchors_file
    assert os.path.exists(anchors_file), "anchor file {} is not exists".format(anchors_file)

    batch_size = int(args.batch_size)
    if batch_size <= 0:
        batch_size = 128

    epochs = int(args.epochs)
    if epochs <= 0:
        epochs = 100

    # todo 加载 class and anchors
    class_names = get_classes(classes_file)
    num_classes = len(class_names)
    anchors = get_anchors(anchors_file)

    # todo model
    # multiple of 32, hw
    input_shape = (416, 416)

    # todo train
    model = create_model(input_shape, anchors, num_classes, freeze_body=2,
                         weights_path=model_file)
    logging = TensorBoard(log_dir=logs_path)
    checkpoint = ModelCheckpoint(logs_path + "{epoch:02d}.h5",
                                 monitor="val_loss",
                                 save_weights_only=True, save_best_only=True, period=3)
    # 更改学习率策略
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=15, verbose=1)

    # 加载数据
    # todo 加载数据
    train_sequence = SequenceData(annotation_train_file, input_shape, anchors, num_classes, batch_size)
    val_sequence = SequenceData(annotation_val_file, input_shape, anchors, num_classes, batch_size)

    # Train with frozen layers first, to get a stable loss.
    # Adjust num epochs to your dataset. This step is enough to obtain a not bad model.
    if True:
        model.compile(optimizer=Adam(lr=1e-3), loss={'yolo_loss': lambda y_true, y_pred: y_pred})
        model.summary()
        model.fit(
            train_sequence,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=[logging, checkpoint],
            validation_data=val_sequence,
            initial_epoch=0,
            steps_per_epoch=train_sequence.get_epochs(),
            validation_steps=val_sequence.get_epochs(),
            # validation_batch_size=batch_size,
            max_queue_size=20,
            workers=4)

        model.save_weights(logs_path + 'trained_weights_stage_1.h5')
        model.save(logs_path + "yolov3-model-1.h5")

    if True:
        for i in range(int(len(model.layers)/2)):
            model.layers[i].trainable = True
        model.compile(optimizer=Adam(lr=1e-4),
                      loss={'yolo_loss': lambda y_true, y_pred: y_pred})  # recompile to apply the change
        print('Unfreeze all of the layers.')

        model.fit(
            train_sequence,
            batch_size=batch_size,
            epochs=epochs*3,
            callbacks=[logging, checkpoint, reduce_lr, early_stopping],
            validation_data=val_sequence,
            initial_epoch=epochs,
            steps_per_epoch=train_sequence.get_epochs(),
            validation_steps=val_sequence.get_epochs(),
            #validation_batch_size=batch_size,
            max_queue_size=20,
            workers=4)

        model.save_weights(logs_path + 'trained_weights_Unfreeze.h5')
        model.save(logs_path + "yolov3-model-2.h5")

if __name__ == "__main__":
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    _main(parse_arguments(sys.argv[1:]))

