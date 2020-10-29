import random
import colorsys
import numpy as np
import cv2

from tensorflow.keras.layers import Input
from PIL import Image

from yolov3.utils import get_classes, get_anchors, letterbox_image
from yolov3.model import yolo_eval, yolo_body


def _main():
    # TODO: 定义路径
    model_path = "logs/yolov3-model-1.h5"
    assert model_path.endswith('.h5'), 'Keras model must be a .h5 file.'
    anchors_path = "model/yolo_anchors.txt"
    classes_path = "model/voc_classes.txt"
    test_path = "images/test1.jpg"
    output_path = "images/test1_out.jpg"

    anchors = get_anchors(anchors_path)
    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    num_anchors = len(anchors)

    # 构建模型
    image_input = Input(shape=(None, None, 3))
    model = yolo_body(image_input, num_anchors // 3, num_classes)
    model.load_weights(model_path)
    model.summary()
    # 给每一个类定义一个颜色
    hsv_tuples = [(x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    random.seed(10101)  # Fixed seed for consistent colors across runs.
    random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    random.seed(None)  # Reset seed to default.
    # TODO: 加载图片
    image = Image.open(test_path)
    print(image.size)
    # 按照原尺寸缩放图像，空余的地方留灰
    boxed_image = letterbox_image(image, (416, 416))
    image_data = np.array(boxed_image, dtype='float32')
    image_data /= 255.
    image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
    # 推理
    y = model.predict(image_data, batch_size=1)
    boxes_, scores_, classes_ = yolo_eval(y, anchors, num_classes, (image.size[1], image.size[0]))
    image = cv2.imread(test_path)
    for i, box in enumerate(boxes_):
        cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), colors[classes_[i]])
        cv2.putText(image, class_names[classes_[i]], (int(box[0]), int(box[1])), 1, 1, colors[classes_[i]], 1)
    cv2.imshow('image', image)
    cv2.imwrite(output_path, image)
    cv2.waitKey(0)


if __name__ == "__main__":
    _main()
