import random
import shutil
import torch
import torch.distributed as dist
import torch.autograd as autograd
from modules.network import get_network
from modules.CONTRIQUE_model import CONTRIQUE_model
from modules.new_caption_img_model import CON_model
from torchvision import transforms
import numpy as np
from tokenizer import SimpleTokenizer
import os
from PIL import Image


def img_feat(model, img_path):

    image = Image.open(img_path).convert('RGB')
    image1 = np.array(image)
    if len(image1.shape) == 2:
        return None

    sz = image.size
    image_2 = image.resize((sz[0] // 2, sz[1] // 2))

    image = random_CerticalHorizon_flip(image)
    image_2 = random_CerticalHorizon_flip(image_2)

    image = transforms.ToTensor()(image).unsqueeze(0).cuda()
    image_2 = transforms.ToTensor()(image_2).unsqueeze(0).cuda()

    caption = 'a photo.'
    tokenizer = SimpleTokenizer()
    caption = tokenizer(caption).cuda(non_blocking=True)
    caption = caption.unsqueeze(0)

    with torch.no_grad():
        _, _, _, _, model_feat, model_feat_2, _, _, _, _ = model(image, image_2, caption)

    feat = np.hstack((model_feat.detach().cpu().numpy(),
                      model_feat_2.detach().cpu().numpy()))

    return feat


def random_CerticalHorizon_flip(img):
    # img = Image.open(img_path)
    one_zero = [0, 1]
    p = random.choice(one_zero)
    p2 = random.choice(one_zero)
    if p == one_zero[0] and p2 == 0:
        p = 1

    pF = transforms.RandomVerticalFlip(p=p2)
    HF = transforms.RandomHorizontalFlip(p=p)  # p为概率，缺省时默认0.5
    hf_image = HF(img)
    hv_image = pF(hf_image)

    return hv_image


def load_SLIQUE_model(model_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    encoder = get_network('resnet50', pretrained=False)
    n_features = encoder.fc.in_features  # get dimensions of fc layer
    c_model = CONTRIQUE_model(None, encoder=encoder, n_features=n_features)
    model = CON_model(embed_dim=512, vision_width=n_features, vision_model=encoder, context_length=77,
                      vocab_size=49408,
                      transformer_width=512, transformer_heads=8, transformer_layers=12, args=None,
                      n_features=n_features,
                      model=c_model)

    model.load_state_dict(torch.load(model_path, map_location=device.type))
    model = model.to(device)
    return model


def get_model(model):
    if isinstance(model, torch.nn.DataParallel) \
      or isinstance(model, torch.nn.parallel.DistributedDataParallel):
        return model.module
    else:
        return model


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


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(state, is_best, output_dir):
    if is_main_process():
        ckpt_path = f'{output_dir}/checkpoint.pt'
        best_path = f'{output_dir}/checkpoint_best.pt'
        torch.save(state, ckpt_path)
        if is_best:
            shutil.copyfile(ckpt_path, best_path)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def scaled_all_reduce(tensors, is_scale=True):
    """Performs the scaled all_reduce operation on the provided tensors.
    The input tensors are modified in-place. Currently supports only the sum
    reduction operator. The reduced values are scaled by the inverse size of the
    world size.
    """
    world_size = get_world_size()
    # There is no need for reduction in the single-proc case
    if world_size == 1:
        return tensors
    # Queue the reductions
    reductions = []
    for tensor in tensors:
        reduction = dist.all_reduce(tensor, async_op=True)
        reductions.append(reduction)
    # Wait for reductions to finish
    for reduction in reductions:
        reduction.wait()
    # Scale the results
    if is_scale:
        for tensor in tensors:
            tensor.mul_(1.0 / world_size)
    return tensors


def all_gather_batch(tensors):
    """
    Performs all_gather operation on the provided tensors.
    """
    # Queue the gathered tensors
    world_size = get_world_size()
    # There is no need for reduction in the single-proc case
    if world_size == 1:
        return tensors
    tensor_list = []
    output_tensor = []
    for tensor in tensors:
        tensor_all = [torch.ones_like(tensor) for _ in range(world_size)]
        dist.all_gather(
            tensor_all,
            tensor,
            async_op=False  # performance opt
        )

        tensor_list.append(tensor_all)

    for tensor_all in tensor_list:
        output_tensor.append(torch.cat(tensor_all, dim=0))
    return output_tensor


class GatherLayer(autograd.Function):
    """
    Gather tensors from all workers with support for backward propagation:
    This implementation does not cut the gradients as torch.distributed.all_gather does.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]


def all_gather_batch_with_grad(tensors):
    """
    Performs all_gather operation on the provided tensors.
    Graph remains connected for backward grad computation.
    """
    # Queue the gathered tensors
    world_size = get_world_size()
    # There is no need for reduction in the single-proc case
    if world_size == 1:
        return tensors
    tensor_list = []
    output_tensor = []

    for tensor in tensors:
        tensor_all = GatherLayer.apply(tensor)
        tensor_list.append(tensor_all)

    for tensor_all in tensor_list:
        output_tensor.append(torch.cat(tensor_all, dim=0))
    return output_tensor


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


# class GaussianBlur(object):
#     """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""
#
#     def __init__(self, sigma=[.1, 2.]):
#         self.sigma = sigma
#
#     def __call__(self, x):
#         sigma = random.uniform(self.sigma[0], self.sigma[1])
#         x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
#         return x
