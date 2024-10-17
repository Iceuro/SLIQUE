import torch
import scipy.io as scio
import csv
import logging
from tqdm import tqdm
from modules.network import get_network
from modules.CONTRIQUE_model import CONTRIQUE_model
from modules.new_caption_img_model import CON_model
from torchvision import transforms
import numpy as np
from tokenizer import SimpleTokenizer
import os
import argparse
import random
from PIL import Image
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

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

def load_CONTRIQUE_model(model_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    encoder = get_network('resnet50', pretrained=False)
    model = CONTRIQUE_model(None, encoder=encoder, n_features=2048)
    model.load_state_dict(torch.load(model_path, map_location=device.type))
    model = model.to(device)
    return model

# img_dir = r'D://database//QADS//super_resolved_images'
img_dir = r'D:\database\GFIQA-20k\image'
# img_dir = r'D://database//SPAQ//TestImage'
# img_dir = r'D:\database\ChallengeDB_release\Images'
# model_path = 'models//checkpoint_25.tar'
# model = load_CONTRIQUE_model(model_path)
# save_dir = 'feat/new_CON/CLIVE'

model_path = 'models//SLIQUE.tar'
model = load_SLIQUE_model(model_path)
save_dir = 'feat/SLIQUE/GFIQA'

names = os.listdir(img_dir)
l = len(names)
mkdir(save_dir)

for i in range(l):
    img_path = os.path.join(img_dir, names[i])
    feat_name = names[i]
    try:
        image = Image.open(img_path).convert('RGB')
    except:
        print("{} is error!".format(names[i]))
        continue
    image1 = np.array(image)
    if len(image1.shape) == 2:
        continue

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

    model.eval()
    #SLIQUE
    with torch.no_grad():
        _, _, _, _, model_feat, model_feat_2, _, _, _, _ = model(image, image_2, caption)
    #CONTRIQUE
    # with torch.no_grad():
    #     _, _, _, _, model_feat, model_feat_2, _, _ = model(image, image_2)

    feat = np.hstack((model_feat.detach().cpu().numpy(),
                      model_feat_2.detach().cpu().numpy()))

    feat_path = os.path.join(save_dir, feat_name)
    np.save(feat_path, feat)
    print('{} is finished!'.format(feat_name))
