import os
import argparse
import time
import os.path as osp

import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torch
from torch import optim as optim
import torch.backends.cudnn as cudnn

from my_dataset import MyDataSet
import model as convnextv2
import utils
from utils import read_split_data, create_lr_scheduler, get_params_groups, train_one_epoch, evaluate, \
    init_distributed_mode, str2bool


def main(args):
    init_distributed_mode(args)
    print(args)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")

    # 设置随机种子，以保证每次运行结果的可重复性
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # 加速卷积神经网络的训练
    cudnn.benchmark = True

    trained_model = args.trained_model
    if os.path.exists(trained_model) is False:
        os.makedirs(trained_model)

    tb_writer = SummaryWriter(log_dir=args.log_dir)

    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)

    img_size = args.img_size
    data_transform = {
        "train": transforms.Compose([transforms.Resize((img_size,img_size)),
                                    transforms.CenterCrop(img_size),
                                    transforms.RandomHorizontalFlip(p=0.5),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        
        "val": transforms.Compose([transforms.Resize((img_size,img_size)),
                                   transforms.CenterCrop(img_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    # 实例化训练数据集
    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"],
                              num_classes=args.num_classes)

    # 实例化验证数据集
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"],
                              num_classes=args.num_classes)

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    sampler_train = torch.utils.data.DistributedSampler(
        train_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True, seed=args.seed,
    )
    print("Sampler_train = %s" % str(sampler_train))

    sampler_val = torch.utils.data.DistributedSampler(
        val_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=False)
    print("Sampler_train = %s" % str(sampler_val))

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               pin_memory=True,
                                               num_workers=nw,
                                               sampler=sampler_train,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             pin_memory=True,
                                             num_workers=nw,
                                             sampler=sampler_val,
                                             collate_fn=val_dataset.collate_fn)

    model = convnextv2.__dict__[args.model](num_classes=args.num_classes).to(device)
    ddp_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_path = args.weights
        weights_dict = torch.load(weights_path, map_location=device)['model_state_dict'] if osp.splitext(weights_path)[1] == '.pth' else torch.load(weights_path, map_location=device)['model']
        # 删除有关分类类别的权重
        for k in list(weights_dict.keys()):
            if "head" in k:
                del weights_dict[k]
        print(ddp_model.module.load_state_dict(weights_dict, strict=False))

    if args.freeze_layers:
        for name, para in ddp_model.module.named_parameters():
            # 除head外，其他权重全部冻结
            if "head" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    # pg = [p for p in model.parameters() if p.requires_grad]
    pg = get_params_groups(ddp_model.module, weight_decay=args.wd)
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=args.wd)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs,
                                       warmup=True, warmup_epochs=1)

    best_acc = 0.
    trained_model_dir = args.trained_model
    os.makedirs(trained_model_dir, exist_ok=True)
    for epoch in range(args.epochs):
        # train
        train_loss, train_acc = train_one_epoch(model=ddp_model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch,
                                                lr_scheduler=lr_scheduler,
                                                args=args)

        # validate
        val_acc = evaluate(model=ddp_model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch)

        tags = ["train_loss", "train_acc", "val_acc", "learning_rate"]
        if utils.is_main_process():
            tb_writer.add_scalar(tags[0], train_loss, epoch)
            tb_writer.add_scalar(tags[1], train_acc, epoch)
            tb_writer.add_scalar(tags[2], val_acc, epoch)
            tb_writer.add_scalar(tags[3], optimizer.param_groups[0]["lr"], epoch)

        if best_acc < val_acc:
            # 保存模型参数和优化器状态到文件
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': ddp_model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'eval_acc': val_acc
            }
            checkpoint_time = time.strftime('%Y%m%d%H%M')
            checkpoint_path = os.path.join(trained_model_dir, f'checkpoint-{checkpoint_time}-{epoch + 1}.pth')
            utils.save_on_master(checkpoint,checkpoint_path)
            best_acc = val_acc

    tb_writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--update_freq', default=1, type=int)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--wd', type=float, default=5e-2)
    parser.add_argument('--model', default='convnextv2_base', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--log_dir', default='/home/11067428/logs')
    parser.add_argument('--seed', default=0, type=int)
    # 数据集所在根目录
    parser.add_argument('--data-path', type=str,default="")
    # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument('--weights', type=str, default='')
    # 是否冻结head以外所有权重
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--trained-model', type=str,default="")
    parser.add_argument('--device', default='cuda')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', type=str2bool, default=False)
    parser.add_argument('--dist_url', default='env://')

    opt = parser.parse_args()

    main(opt)