import os
import shutil

image_label_dist = {}
# 读取图片分类文件并处理图片
image_label_file = open("/data/juicefs_sharing_data/11067428/data/20231116-vulgar/label_datasets_res_all.txt", "r", encoding="utf-8")
for line in image_label_file:
    line = line.strip()
    image_name, label_id = line.split("|")
    image_label_id = image_label_dist.get(image_name, '')

    if not image_label_id:
        image_label_id = label_id
    else:
        image_label_id += ' {}'.format(label_id)
    image_label_dist[image_name] = image_label_id
image_label_file.close()

# 将处理后的数据写入到文件中
output_file = '/data/juicefs_sharing_data/11067428/data/20231116-vulgar/new_label_datasets_res_all.txt'
with open(output_file, 'w') as f:
    for image_name, image_label_id in image_label_dist.items():
        f.write('{}|{}\n'.format(image_name, image_label_id))