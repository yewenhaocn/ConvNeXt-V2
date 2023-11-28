import os
import shutil
# 从文件夹pic中读取所有图片
image_path = '/data/juicefs_sharing_data/11067428/data/test_train/pic'
# 随机取500张图片，复制到test_train目录下
images = [os.path.join(image_path, i) for i in os.listdir(image_path)][:500]
test_train_path = '/data/juicefs_sharing_data/11067428/data/test_train/pic2'
for image in images:
    img_name = image.split("/")[-1]
    shutil.copy(image, os.path.join(test_train_path, img_name))