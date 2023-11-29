import argparse
import os
import sys
import json
import pickle
import random
import math
from PIL import Image

import torch.distributed as dist

import torch
from tqdm import tqdm


def read_split_data(root: str, val_rate: float = 0.2):
    random.seed(0)  # 保证随机结果可复现
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    # 从文件label_name.txt中获取所有的类别
    label_file = open(os.path.join(root, "label_name.txt"), "r", encoding="utf-8")
    image_label_dict = {}
    for line in label_file:
        line = line.strip()
        label_id, label_name = line.split("|")
        label_id = int(label_id)  # 将label_id转换为整数
        image_label_dict[label_id] = label_name
    label_file.close()
    image_label_list = list(image_label_dict.keys())
    # 排序，保证个平台顺序一致
    image_label_list.sort()
    # 生成类别名称和id的关系文件
    json_str = json.dumps(dict((key, val) for key, val in image_label_dict.items()), indent=4)
    with open(os.path.join(root, "class_indices.json"), 'w') as json_file:
        json_file.write(json_str)
    # 给label_id维护索引
    image_label_id_index_dict = {i: value for i, value in enumerate(image_label_list)}
    label_id_index_json = json.dumps(dict((key, val) for key, val in image_label_id_index_dict.items()), indent=4)
    with open(os.path.join(root, "class_indices_index.json"), 'w') as label_id_index_json_file:
        label_id_index_json_file.write(label_id_index_json)

    train_images_path = []  # 存储训练集的所有图片路径
    train_images_label = []  # 存储训练集图片对应索引信息
    val_images_path = []  # 存储验证集的所有图片路径
    val_images_label = []  # 存储验证集图片对应索引信息

    # 从文件new_label_datasets_res_all.txt中获取每个文件对应的类型
    image_label_file = open(os.path.join(root, "new_label_datasets_res_all.txt"), "r", encoding="utf-8")
    image_label_rela_dist = {}
    for line in image_label_file:
        line = line.strip()
        image_name, label_id = line.split("|")
        label_id_list = list(map(int, label_id.split()))
        # 需要将label_id转成对应的索引
        label_id_index_list = [index for index, label_id in zip(image_label_id_index_dict.keys(), image_label_id_index_dict.values()) if label_id in label_id_list]
        image_label_rela_dist[image_name] = label_id_index_list
    image_label_file.close()

    # 从文件夹pic中读取所有图片
    image_path = os.path.join(root, "pic")

    valid_images = []
    images = [os.path.join(image_path, i) for i in os.listdir(image_path)]
    for image in images:
        image_name = image.split("/")[-1]
        # 过滤掉没有label的图片
        if image_name not in image_label_rela_dist:
            continue
        # 过滤掉有问题的图片
        try:
            img = Image.open(image).convert('RGB')
        except Exception as e:
            print(f"Cannot identify image file {image}. Skipping...,exception is {e}")
            continue
        #校验通过的图片
        valid_images.append(image)

    # 排序，保证各平台顺序一致
    valid_images.sort()
    # 按比例随机采样验证样本
    val_path = random.sample(valid_images, k=int(len(valid_images) * val_rate))

    for img_path in valid_images:
        if img_path in val_path:  # 如果该路径在采样的验证集样本中则存入验证集
            val_images_path.append(img_path)
            # 获取改图片对应的分类
            img_name = img_path.split("/")[-1]
            val_images_label.append(image_label_rela_dist.get(img_name))
        else:  # 否则存入训练集
            train_images_path.append(img_path)
            # 获取改图片对应的分类
            img_name = img_path.split("/")[-1]
            train_images_label.append(image_label_rela_dist.get(img_name))


    print("{} images were found in the dataset.".format(len(valid_images)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))
    assert len(train_images_path) > 0, "number of training images must greater than 0."
    assert len(val_images_path) > 0, "number of validation images must greater than 0."
    return train_images_path, train_images_label, val_images_path, val_images_label


def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def is_main_process():
    return get_rank() == 0

def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def train_one_epoch(model, optimizer, data_loader, device, epoch, lr_scheduler, args=None):
    data_loader.sampler.set_epoch(epoch)
    model.train()
    loss_function = torch.nn.MultiLabelSoftMarginLoss()

    # 在分布式训练中，需要使用all_reduce来合并各个进程的损失和准确度
    # 初始化全局损失和准确度的tensor
    accu_loss = torch.tensor(0.0).to(device)  # 累计损失
    accu_num = torch.tensor(0.0).to(device)  # 累计预测正确的样本数
    sample_num = torch.tensor(0.0).to(device)  # 样本数量

    optimizer.zero_grad()

    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += torch.tensor(images.shape[0]).to(device)

        pred = model(images.to(device))

        labels = labels.to(device).float()
        # labels = torch.repeat_interleave(labels, labels.shape[1], dim=1)
        accu_num += torch.eq((pred > 0.5).float(), labels).sum()

        loss = loss_function(pred, labels)
        loss.backward()
        accu_loss += loss.detach()

        # 同步操作，确保所有进程都完成了梯度计算
        dist.barrier()

        # 使用all_reduce进行全局求和
        dist.all_reduce(accu_loss)
        dist.all_reduce(accu_num)
        dist.all_reduce(sample_num)

        # 将全局统计信息转换为Python标量，并计算平均值
        accu_loss = accu_loss.item() / dist.get_world_size()
        accu_num = accu_num.item() / dist.get_world_size()
        sample_num = sample_num.item() / dist.get_world_size()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}, lr: {:.5f}".format(
            epoch,
            accu_loss / (step + 1),
            accu_num / sample_num,
            optimizer.param_groups[0]["lr"]
        )

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()
        # update lr
        if step % args.update_freq == 0:
            lr_scheduler.step()

    return accu_loss / (step + 1), accu_num / sample_num


@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    data_loader.sampler.set_epoch(epoch)
    loss_function = torch.nn.MultiLabelSoftMarginLoss()

    model.eval()

    accu_loss = torch.tensor(0.0).to(device)  # 累计损失
    accu_num = torch.tensor(0.0).to(device)  # 累计预测正确的样本数
    sample_num = torch.tensor(0.0).to(device)  # 样本数量

    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += torch.tensor(images.shape[0]).to(device)

        pred = model(images.to(device))

        labels = labels.to(device).float()
        # labels = torch.repeat_interleave(labels, labels.shape[1], dim=1)
        accu_num += torch.eq((pred > 0.5).float(), labels).sum()

        loss = loss_function(pred, labels)
        accu_loss += loss

        # 同步操作，确保所有进程都完成了梯度计算
        dist.barrier()

        # 使用all_reduce进行全局求和
        dist.all_reduce(accu_loss)
        dist.all_reduce(accu_num)
        dist.all_reduce(sample_num)

        # 将全局统计信息转换为Python标量，并计算平均值
        world_size = dist.get_world_size()
        accu_loss = accu_loss.item() / world_size
        accu_num = accu_num.item() / world_size
        sample_num = sample_num.item() / world_size

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(
            epoch,
            accu_loss / (step + 1),
            accu_num / sample_num
        )

    return accu_loss / (step + 1), accu_num / sample_num


def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3,
                        end_factor=1e-6):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            current_step = (x - warmup_epochs * num_step)
            cosine_steps = (epochs - warmup_epochs) * num_step
            # warmup后lr倍率因子从1 -> end_factor
            return ((1 + math.cos(current_step * math.pi / cosine_steps)) / 2) * (1 - end_factor) + end_factor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)



def get_params_groups(model: torch.nn.Module, weight_decay: float = 1e-5):
    # 记录optimize要训练的权重参数
    parameter_group_vars = {"decay": {"params": [], "weight_decay": weight_decay},
                            "no_decay": {"params": [], "weight_decay": 0.}}

    # 记录对应的权重名称
    parameter_group_names = {"decay": {"params": [], "weight_decay": weight_decay},
                             "no_decay": {"params": [], "weight_decay": 0.}}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights

        if len(param.shape) == 1 or name.endswith(".bias"):
            group_name = "no_decay"
        else:
            group_name = "decay"

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)

    print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())

def init_distributed_mode(args):

    if args.dist_on_itp:
        args.rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        args.world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        args.gpu = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        args.dist_url = "tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
        os.environ['LOCAL_RANK'] = str(args.gpu)
        os.environ['RANK'] = str(args.rank)
        os.environ['WORLD_SIZE'] = str(args.world_size)
        # ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()

        os.environ['RANK'] = str(args.rank)
        os.environ['LOCAL_RANK'] = str(args.gpu)
        os.environ['WORLD_SIZE'] = str(args.world_size)
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(
        args.rank, args.dist_url, args.gpu), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def str2bool(v):
    """
    Converts string to bool type; enables command line
    arguments in the format of '--arg1 true --arg2 false'
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
