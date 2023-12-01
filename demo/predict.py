import json
import argparse

import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import model as convnextv2
import os
from pandas.core.frame import DataFrame

def main(opt):
    device = torch.device(opt.device if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")
    img_size = opt.img_size
    data_transform = transforms.Compose(
        [transforms.Resize(img_size),
         transforms.CenterCrop(img_size),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # 读取分类id和分类名称映射关系
    class_indict = {}
    class_path = opt.class_path
    assert os.path.exists(class_path), "file: '{}' dose not exist.".format(class_path)
    with open(class_path, "r") as f:
        class_indict = json.load(f)

    # 读取分类id和分类id索引映射关系
    class_id_mapping_indict = {}
    class_id_mapping_path = opt.class_id_mapping_path
    assert os.path.exists(class_id_mapping_path), "file: '{}' dose not exist.".format(class_id_mapping_path)
    with open(class_id_mapping_path, 'r') as f:
        class_id_mapping_indict = json.load(f)

    # 加载图片
    clas = []
    pathss = []
    model = convnextv2.__dict__[opt.model](num_classes=opt.num_classes).to(device)
    model_weight_path = opt.weights
    model.load_state_dict(torch.load(model_weight_path, map_location=device)['model_state_dict'])
    img_dir = opt.img_dir
    paths = os.listdir(img_dir)
    for i in tqdm(range(len(paths))):
        img_path = os.path.join(img_dir,paths[i])
        assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
        img = Image.open(img_path).convert('RGB')
        pathss.append(paths[i])

        # [N, C, H, W]
        img = data_transform(img)
        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)
        model.eval()
        with torch.no_grad():
            # predict class
            output = torch.squeeze(model(img.to(device))).cpu()
            threshold = 0.5
            softmax_output  = torch.sigmoid(output)
            indices_values = dict(filter(lambda x: x[1] > threshold, enumerate(softmax_output)))
            idxs = (softmax_output > threshold).nonzero().flatten()

        #clas.append(",".join([class_indict[str(idx.item())][:-1] for idx in idxs]))
    c={"a" : pathss,"b" : clas}#将列表a，b转换成字典
    data=DataFrame(c)#将字典转换成为数据框
    data.to_csv(opt.output_path,sep=',',index=False,header=False)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=51)
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--model', default='convnextv2_base', type=str, metavar='MODEL')
    # 生成分类对应关系的目录
    parser.add_argument('--class_path', type=str, default="")
    parser.add_argument('--class_id_mapping_path', type=str, default="")
    # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument('--weights', type=str, default='')
    # 需要推理的图片目录
    parser.add_argument("--img_dir", type=str, default='')
    # 推理结果目录
    parser.add_argument("--output_path", type=str,default='')
    parser.add_argument('--device', default='cuda')

    opt = parser.parse_args()
    main(opt)