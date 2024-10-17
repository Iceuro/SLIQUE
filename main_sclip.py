import numpy as np
import torch
import argparse
import os
import pandas as pd
from sklearn.utils import shuffle
from torch import nn
import clip
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torchvision.transforms as transforms
import timm
import math
import sys
# distributed training
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP
from modules.dataset_loader import image_caption_data
from modules.network import get_network
from modules.CONTRIQUE_model import CONTRIQUE_model
from modules.new_caption_img_model import CON_model, CLIP
from modules.nt_xent_multiclass import NT_Xent
from modules.configure_optimizers import configure_optimizers
from tokenizer import SimpleTokenizer
from model_io import save_model
from modules.sync_batchnorm import convert_model
# from modules.tokenizer import tokenize
import time
import datetime
from PIL import ImageFile
# import tokenizer.tokenize
# os.environ["CUDA_VISIBLE_DEVICES"]="7"
ImageFile.LOAD_TRUNCATED_IMAGES = True
# find_unused_parameters=True
torch.multiprocessing.set_sharing_strategy('file_system')
# import random
def mixgen_batch(image, text,  tokenizer,lam=0.5):
    batch_size = image.size()[0]
    index = np.random.permutation(batch_size)
    text_list = list(text)
    # tranform_toT = transforms.Compose([
    #     # transforms.RandomHorizontalFlip(0.5),
    #     transforms.ToTensor(),
    #
    # ])
    for i in range(batch_size):
        # image mixup
        image[i,:] = lam * image[i,:] + (1 - lam) * image[index[i],:]
        # text concat
        text_list[i] = text_list[i] + " " + text_list[index[i]]

    # image = tranform_toT(image)
    text = tuple(text_list)
    text = tokenizer(text)
    return image,text

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
def clip_loss(x_i1,logits_per_image,logits_per_text,args):
    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()
    ground_truth = torch.arange(len(x_i1), dtype=torch.long, device=args.device)
    loss2 = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2

    # with torch.no_grad():
    #     pred = torch.argmax(logits_per_image, dim=-1)
    #     correct = pred.eq(ground_truth).sum()
    #     acc = 100 * correct / len(x_i1)

    return {'loss': loss2, 'clip_loss': loss2}
def opt(args,model):
    def exclude_from_wd_and_adaptation(name):
        if 'bn' in name:
            return True
    param_groups = [
        {
            'params': [p for name, p in model.named_parameters() if not exclude_from_wd_and_adaptation(name)],
            'weight_decay': args.weight_decay,
            'layer_adaptation': True,
        },
        {
            'params': [p for name, p in model.named_parameters() if exclude_from_wd_and_adaptation(name)],
            'weight_decay': 0.,
            'layer_adaptation': False,
        },
    ]

    LR = args.lr

    if args.opt == 'sgd':
        optimizer = torch.optim.SGD(
            param_groups,
            lr=LR,
            momentum=0.9,
        )
    elif args.opt == 'adam':
        optimizer = torch.optim.Adam(
            param_groups,
            lr=LR,
        )
    else:
        raise NotImplementedError
    return optimizer


def train(args, train_loader_syn, train_loader_ugc, \
          model, criterion, optimizer, scaler, tokenize,scheduler=None):
    loss_epoch = 0
    model.train()

    for step, ((syn_clip_i,syn_i1, syn_i2, dist_label_syn,y_i), (ugc_clip_j,ugc_i1, ugc_i2,_ ,y_j)) in \
            enumerate(zip(train_loader_syn, train_loader_ugc)):

        # image 1
        syn_i1 = syn_i1.cuda(non_blocking=True)
        ugc_i1 = ugc_i1.cuda(non_blocking=True)
        x_i1 = torch.cat((syn_i1, ugc_i1), dim=0)

        # image 2
        syn_i2 = syn_i2.cuda(non_blocking=True)
        ugc_i2 = ugc_i2.cuda(non_blocking=True)
        x_i2 = torch.cat((syn_i2, ugc_i2), dim=0)


        # x_i3 = torch.cat((x_i1, x_i2), dim=0)
        syn_clip_i = syn_clip_i.cuda(non_blocking=True)
        ugc_clip_j = ugc_clip_j.cuda(non_blocking=True)
        clip_i=torch.cat((syn_clip_i, ugc_clip_j), dim=0)


        y_i = y_i.cuda(non_blocking=True)
        y_j = y_j.cuda(non_blocking=True)
        y_i1=torch.cat((y_i, y_j), dim=0)

        dist_label = torch.zeros((2 * args.batch_size,
                                  args.clusters + (args.batch_size * args.nodes)))

        dist_label[:args.batch_size, :args.clusters] = dist_label_syn.clone()
        dist_label[args.batch_size:, args.clusters + (args.nr * args.batch_size): \
                                     args.clusters + ((args.nr + 1) * args.batch_size)] = \
            torch.eye(args.batch_size)

        # all local patches inherit class of the orginal image
        dist_label = dist_label.repeat(1, args.num_patches).view(-1, dist_label.shape[1])
        dist_label = dist_label.cuda(non_blocking=True)


        with torch.cuda.amp.autocast(enabled=True):
            _, _, z_i1_patch, z_i2_patch, _, _, _, _ ,logits_per_image, logits_per_text\
                = model(x_i1, x_i2, y_i1)
            # logits_per_image, logits_per_text\
            #     = model(x_i1, x_i2,y_i1)
            loss1 = criterion(z_i1_patch, z_i2_patch, dist_label)
            clip_loss_stat = clip_loss(x_i1, logits_per_image, logits_per_text,args)
            loss2 = clip_loss_stat['loss']
            # loss=loss2
            loss = (0.7*loss1+0.3*loss2)
            # loss=(0.6*loss1+0.4*loss2)

        # update model weights
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        # optimizer.step()
        scaler.update()

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            sys.exit(1)

        if dist.is_available() and dist.is_initialized():
            loss = loss.data.clone()
            dist.all_reduce(loss.div_(dist.get_world_size()))

        if args.nr == 0 and step % 5 == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(f"Step [{step}/{args.steps}]\t Loss: {loss.item()}\t con_Loss1: {loss1.item()}\t text_Loss2: {loss2.item()}\t LR: {round(lr, 5)}")
            # print(f"Step [{step}/{args.steps}]\t Loss: {loss.item()}\t text_Loss2: {loss2.item()}\t LR: {round(lr, 5)}")
            # print(f"Step [{step}/{args.steps}]\t Loss: {loss.item()}\t con_Loss1: {loss1.item()}\t  LR: {round(lr, 5)}")

        if args.nr == 0:
            args.global_step += 1

        loss_epoch += loss.item()
        # optimizer.step()
        if scheduler:
            scheduler.step()

    return loss_epoch





def main(gpu, args):
    rank = args.nr * args.gpus + gpu
    if args.nodes > 1:
        cur_dir = 'file://' + os.getcwd() + '/sharedfile'
        dist.init_process_group("nccl", init_method=cur_dir, \
                                rank=rank, timeout=datetime.timedelta(seconds=3600), \
                                world_size=args.world_size)
        # dist.init_process_group("nccl",
        #                         rank=rank,
        #                         world_size=args.world_size)
        torch.cuda.set_device(gpu)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    tokenizer =SimpleTokenizer()
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])
    # train_transform = transforms.Compose([
    #     transforms.RandomResizedCrop(256, scale=(0.5, 1.0)),
    #     transforms.ToTensor(),
    #     normalize
    # ])
    # loader for synthetic distortions data
    fls = pd.read_csv(args.csv_file_syn, low_memory=False,encoding='utf_8_sig')
    # fls = shuffle(fls)
    train_dataset_syn = image_caption_data(file_path=args.PATH_file_syn,fls=fls, tokenizer=tokenizer,data_type='syn'
                                   ,image_size=args.image_size)

    if args.nodes > 1:
        train_sampler_syn = torch.utils.data.distributed.DistributedSampler(
            train_dataset_syn, num_replicas=args.world_size, rank=rank, shuffle=True
        )
    else:
        train_sampler_syn = None

    train_loader_syn = torch.utils.data.DataLoader(
        train_dataset_syn,
        batch_size=args.batch_size,
        shuffle=(train_sampler_syn is None),
        drop_last=True,
        num_workers=args.workers,
        sampler=train_sampler_syn,
    )

    # loader for authetically distorted data
    fls1 = pd.read_csv(args.csv_file_ugc, low_memory=False,encoding='utf_8_sig')
    # fls1 = shuffle(fls1)
    train_dataset_ugc = image_caption_data(file_path=args.PATH_file_ugc,fls=fls1, tokenizer=tokenizer,data_type='ugc',
                                   image_size=args.image_size)

    if args.nodes > 1:
        train_sampler_ugc = torch.utils.data.distributed.DistributedSampler(
            train_dataset_ugc, num_replicas=args.world_size, rank=rank, shuffle=True
        )
    else:
        train_sampler_ugc = None

    train_loader_ugc = torch.utils.data.DataLoader(
        train_dataset_ugc,
        batch_size=args.batch_size,
        shuffle=(train_sampler_ugc is None),
        drop_last=True,
        num_workers=args.workers,
        sampler=train_sampler_ugc,
    )

    # initialize ResNet
    encoder = get_network(args.network, pretrained=False)
    encoder =  timm.create_model('resnet50', pretrained=False, num_classes=0)
    args.n_features = 2048  # get dimensions of fc layer
    c_model = CONTRIQUE_model(args=args, encoder=encoder,
                      n_features=args.n_features)

    # model=model()
    # initialize model
    # if args.reload:
    model_fp = os.path.join(
        args.contrast_model_path, "CONTRIQUE_checkpoint{}.tar".format(args.epoch_num)
    )
    c_model.load_state_dict(torch.load(model_fp, map_location=args.device.type))
    c_model = c_model.to(args.device)

    # initialize model
    # vision_model = timm.create_model('resnet50', pretrained=False, num_classes=0)
    # clip_model, preprocess = clip.load("ViT-B/32", device=args.device, jit=False)

    model = CON_model(args=args, embed_dim=512, vision_width=args.n_features, vision_model=encoder, context_length=77,
                      vocab_size=49408,
                      transformer_width=512, transformer_heads=8, transformer_layers=12, n_features=args.n_features,
                      model=c_model)

    model.load_state_dict(torch.load('/raid/hzc/py/iqa_project/syn_models_all/UGC_SYN_7.3_7V3loss_autn4metric_text_quality/new_checkpoint_9.tar', map_location=args.device.type))
    model = model.to(args.device)
    # sgd optmizer
    args.steps = min(len(train_loader_syn), len(train_loader_ugc))


    args.lr_schedule = 'warmup-anneal'
    args.warmup = 0.1
    args.weight_decay = 1e-4
    args.iters = args.steps * args.epochs
    optimizer = opt(args, model)
    scheduler = configure_optimizers(args, optimizer, cur_iter=-1)

    criterion = NT_Xent(args.batch_size, args.temperature, args.device, args.world_size)
    # device_ids=[0,1]
    # DDP / DP
    if args.dataparallel:
        model = convert_model(model)
        model = DataParallel(model)

    else:
        if args.nodes > 1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = DDP(model, device_ids=[gpu],find_unused_parameters=True)
            print(rank)
            dist.barrier()

    model = model.to(args.device)

    scaler = torch.cuda.amp.GradScaler(enabled=True)

    #    writer = None
    if args.nr == 0:
        print('Training Started')

    if not os.path.isdir(args.model_path):
        os.mkdir(args.model_path)

    epoch_losses = []
    args.global_step = 0
    args.current_epoch = args.start_epoch
    for epoch in range(args.start_epoch, args.epochs):
        start = time.time()

        loss_epoch = train(args, train_loader_syn, train_loader_ugc,
                           model, criterion, optimizer, scaler, tokenizer,scheduler)

        end = time.time()
        print(np.round(end - start, 4))

        if args.nr == 0 and epoch % 1 == 0:
            save_model(args, model, optimizer)
            torch.save({'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict()},
                       args.model_path + 'optimizer.tar')
        #
        if args.nr == 0:
            print(
                f"Epoch [{epoch}/{args.epochs}]\t Loss: {loss_epoch/args.steps }\t steps:{args.steps }"
            )
            args.current_epoch += 1
            epoch_losses.append(loss_epoch/args.steps)
            np.save(args.model_path + 'losses.npy', epoch_losses)

    ## end training
    save_model(args, model, optimizer)
    # torch.save({'optimizer': optimizer.state_dict(),
    #             'scheduler': scheduler.state_dict()},
    #            args.model_path + 'optimizer.tar')

def parse_args():
    parser = argparse.ArgumentParser(description="CONTRIQUE")
    parser.add_argument('--nodes', type=int, default=1, help='number of nodes', metavar='')
    parser.add_argument('--nr', type=int, default=0, help='rank', metavar='')
    parser.add_argument('--weight', type=int, default=3, help='contrique weight')
    # parser.add_argument('--model', default='RES50', type=str)
    parser.add_argument('--csv_file_syn', type=str,
                        default=r'csv_files/refresh//syn.csv',
                        help='list of filenames of images with synthetic distortions')
    parser.add_argument('--PATH_file_syn', type=str,
                        default=r'F://syn',
                        help='PATH of filenames of SYN images')
    parser.add_argument('--csv_file_ugc', type=str,
                        default=r'csv_files/refresh/ugc.csv',
                        help='list of filenames of UGC images')
    parser.add_argument('--PATH_file_ugc', type=str,
                        default='F://UGC_data',
                        help='PATH of filenames of UGC images')

    parser.add_argument('--image_size', type=tuple, default=(256, 256),
                        help='image size')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='number of images in a batch')
    parser.add_argument('--workers', type=int, default=8,
                        help='number of workers')
    parser.add_argument('--opt', type=str, default='sgd',
                        help='optimizer type')
    parser.add_argument('--lr', type=float, default=0.006,
                        help='learning rate')
    parser.add_argument('--network', type=str, default='resnet50',
                        help='network architecture')
    parser.add_argument('--model_path', type=str, default='./syn_models_all/UGC_SYN_7.3_7V3loss_autn4metric_text_quality/',
                        help='folder to save trained models')
    parser.add_argument('--contrast_model_path', type=str, default='./model_path models/',
                        help='folder to save trained models')
    parser.add_argument('--temperature', type=float, default=0.1,
                        help='temperature parameter')
    parser.add_argument('--clusters', type=int, default=126,
                        help='number of synthetic distortion classes')
    parser.add_argument('--reload', type=bool, default=True,
                        help='reload trained model')
    parser.add_argument('--normalize', type=bool, default=True,
                        help='normalize encoder output')
    parser.add_argument('--patch_dim', type=tuple, default=(2, 2),
                        help='number of patches for each input image')
    parser.add_argument('--projection_dim', type=int, default=128,
                        help='dimensions of the output feature from projector')
    parser.add_argument('--dataparallel', type=bool, default=False,
                        help='use dataparallel module of PyTorch')
    parser.add_argument('--start_epoch', type=int, default=10,
                        help='starting epoch number')
    parser.add_argument('--epochs', type=int, default=25,
                        help='total number of epochs')
    # parser.add_argument('--model_path', type=str, \
    #                     default='model_path models/CONTRIQUE.tar', \
    #                     help='Path to trained CONTRIQUE model', metavar='')
    parser.add_argument('--epoch_num', type=int, default=25
                    )
    parser.add_argument('--seed', type=int, default=10,
                        help='random seed')
    parser.add_argument('--master_prot', type=str, default=10,
                        help='random seed')
    args = parser.parse_args()
    mkdir(args.model_path)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "8012"
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.num_gpus = torch.cuda.device_count()
    args.gpus = 1
    args.world_size = args.gpus * args.nodes
    args.num_patches = args.patch_dim[0] * args.patch_dim[1]
    return args


if __name__ == "__main__":
    args = parse_args()

    if args.nodes > 1:
        print(
            f"Training with {args.nodes} nodes, waiting until all nodes join before starting training"
        )
        mp.spawn(main, args=(args,), nprocs=args.gpus, join=True)
    else:
        main(0, args)
