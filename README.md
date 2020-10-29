# 参考 https://github.com/qqwweee/keras-yolo3 的代码，将其中tf1的部分改写为tf2
# 1.模型转换
	下载yolov3-voc.cfg和yolov3.weights 放到和convert.py相同目录
	转换模型
	python convert.py yolov3.cfg yolov3.weights model/yolov3.h5
	
# 2.生成数据
	python voc_annotation.py voc_path
	比如你的目录是/home/ubuntu/work/dataset/Pascal-VOC/VOCdevkit，那么
	voc_path 为 /home/ubuntu/work/dataset/Pascal-VOC
# 3.训练模型
	python train.py 2007_train.txt 2007_val.txt model/yolo.h5 --batch_size 8 --epochs 20

	2007_train.txt 训练的数据文件
	2007_val.txt 	验证的数据文件
	model/yolo.h5 	预训练模型
	--batch_size 8 	一个批次大小
	--epochs 20		训练epochs
	
# 4.模型测试
	python yolo_test.py
	