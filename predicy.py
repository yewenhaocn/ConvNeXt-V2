import torch.utils.data.distributed
from PIL import Image
from torch.autograd import Variable
from datasets import build_transform
from utils import str2bool
import os
import argparse
import models.convnextv2 as convnextv2

classes = ('封面低俗_性感_低俗着装_黑丝', '封面低俗_性感_低俗着装_超短裙', '封面低俗_性感_低俗着装_露背装', '封面低俗_性感_低俗着装_网状袜', '封面低俗_性感_低俗着装_真空类', '封面低俗_性感_低俗着装_热裤', '封面低俗_性感_低俗着装_内衣或泳衣', '封面低俗_性感_胸部相关_画面聚焦胸部', '封面低俗_性感_第二性征_画面聚焦臀部', '封面低俗_性感_第二性征_画面聚焦足部', '封面低俗_性感_第二性征_画面聚焦锁骨', '封面低俗_性感_第二性征_画面聚焦腰部', '封面低俗_性感_第二性征_大腿根部', '封面低俗_性感_第二性征_人鱼线', '封面低俗_性感_第二性征_骆驼趾', '封面低俗_性感_胸部相关_乳沟', '封面低俗_性感_第二性征_屁股沟', '封面低俗_暧昧画面_亲热戏_舔耳垂', '封面低俗_暧昧画面_亲热戏_吻脖子', '封面低俗_暧昧画面_亲热戏_摸胸', '封面低俗_暧昧画面_亲热戏_脱衣', '封面低俗_暧昧画面_亲热戏_咬下巴', '封面低俗_暧昧画面_亲热戏_摸大腿', '封面低俗_暧昧画面_亲热戏_摸臀', '封面低俗_暧昧画面_亲热戏_强奸画面', '封面低俗_暧昧画面_吻戏_亲吻', '封面低俗_不雅动作_低俗手势_竖中指', '封面低俗_不雅动作_性联想姿势_舔唇', '封面低俗_不雅动作_性联想姿势_高潮脸', '封面低俗_性器官物化_男性器官', '封面低俗_性器官物化_女性器官', '封面低俗_成人用品_情趣用品_项圈', '封面低俗_成人用品_情趣用品_润滑液', '封面低俗_成人用品_情趣用品_绳子（捆绑）', '封面低俗_成人用品_情趣用品_皮鞭', '封面低俗_成人用品_情趣用品_丝袜', '封面低俗_成人用品_情趣用品_肛塞', '封面低俗_成人用品_情趣用品_开裆衣', '封面低俗_成人用品_情趣用品_丁字裤', '封面低俗_低俗行业_三级片影星', '封面低俗_低俗行业_低俗场所', '封面低俗_字幕低俗_低俗文字', '封面低俗_性教育_性知识', '封面低俗_性教育_性病', '封面低俗_暧昧画面_亲热戏_其他类别', '正常', '封面低俗_不雅动作_低俗手势_其他低俗手势', '封面低俗_不雅动作_性联想姿势_M字腿', '封面低俗_性感_胸部相关_胸部裸露', '封面低俗_不雅动作_性联想姿势_其他低俗姿势')

parser = argparse.ArgumentParser('predicy', add_help=False)
parser.add_argument('--input_size', default=224, type=int,
                        help='image input size')
parser.add_argument('--crop_pct', type=float, default=None)
parser.add_argument('--imagenet_default_mean_and_std', type=str2bool, default=True)
args = parser.parse_args()
transform_test = build_transform(is_train=False,args=args)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = convnextv2.__dict__['convnextv2_base']()
#checkpoint = torch.load('/data/juicefs_sharing_data/11067428/save_results/checkpoint-0.pth', map_location='cuda')
#model.load_state_dict(checkpoint,strict=False)
model.eval()
model.to(DEVICE)

path = '/data/juicefs_sharing_data/11067428/test/'
testList = os.listdir(path)
for file in testList:
    img = Image.open(path + file)
    img = transform_test(img)
    img.unsqueeze_(0)
    img = Variable(img).to(DEVICE)
    out = model(img)
    # Predict
    _, pred = torch.max(out.data, 1)
    print('Image Name:{},predict:{}'.format(file, classes[pred.data.item()]))
