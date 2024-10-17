import json
import pandas as pd
from torch.utils.data import Dataset
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import os
from PIL import ImageCms
from skvideo.utils.mscn import gen_gauss_window
import scipy.ndimage


def mixgen(image, text, num, lam=0.5):
    # default MixGen
    for i in range(num):
        # image mixup
        image[i, :] = lam * image[i, :] + (1 - lam) * image[i + num, :]
        # text concat
        text[i] = text[i] + " " + text[i + num]
    return image, text


def mixgen_batch(image, text, lam=0.5):
    batch_size = image.size()[0]
    index = np.random.permutation(batch_size)
    for i in range(batch_size):
        # image mixup
        image[i, :] = lam * image[i, :] + (1 - lam) * image[index[i], :]
        # text concat
        text[i] = text[i] + " " + text[index[i]]
    return image, text


def random_CerticalHorizon_flip(num, img):
    # img = Image.open(img_path)
    # one_zero = [0, 1]
    # p = random.choice(one_zero)
    # p2 = random.choice(one_zero)
    # if p == one_zero[0] and p2 == 0:
    if num == 0:
        hf_image = img
    else:
        p = 1

        # pF = transforms.RandomVerticalFlip(p=p2)
        HF = transforms.RandomHorizontalFlip(p=p)  # p为概率，缺省时默认0.5
        hf_image = HF(img)
    # hv_image = pF(hf_image)

    return hf_image


def ResizeCrop(image, sz, div_factor):
    image_size = image.size
    image = transforms.Resize([image_size[1] // div_factor, \
                               image_size[0] // div_factor])(image)
    # image = transforms.Resize([image_size[1] , \
    #                                image_size[0]])(image)

    if image.size[1] < sz[0] or image.size[0] < sz[1]:
        # image size smaller than crop size, zero pad to have same size
        image = transforms.CenterCrop(sz)(image)
    else:
        image = transforms.RandomCrop(sz)(image)

    return image


def compute_MS_transform(image, window, extend_mode='reflect'):
    h, w = image.shape
    mu_image = np.zeros((h, w), dtype=np.float32)
    scipy.ndimage.correlate1d(image, window, 0, mu_image, mode=extend_mode)
    scipy.ndimage.correlate1d(mu_image, window, 1, mu_image, mode=extend_mode)
    return image - mu_image


def MS_transform(image):
    #   MS Transform
    image = np.array(image).astype(np.float32)
    window = gen_gauss_window(3, 7 / 6)
    image[:, :, 0] = compute_MS_transform(image[:, :, 0], window)
    image[:, :, 0] = (image[:, :, 0] - np.min(image[:, :, 0])) / (np.ptp(image[:, :, 0]) + 1e-3)
    image[:, :, 1] = compute_MS_transform(image[:, :, 1], window)
    image[:, :, 1] = (image[:, :, 1] - np.min(image[:, :, 1])) / (np.ptp(image[:, :, 1]) + 1e-3)
    image[:, :, 2] = compute_MS_transform(image[:, :, 2], window)
    image[:, :, 2] = (image[:, :, 2] - np.min(image[:, :, 2])) / (np.ptp(image[:, :, 2]) + 1e-3)

    image = Image.fromarray((image * 255).astype(np.uint8))
    return image


def colorspaces(im, val):
    if val == 0:
        im = transforms.RandomGrayscale(p=1.0)(im)
    elif val == 1:
        srgb_p = ImageCms.createProfile("sRGB")
        lab_p = ImageCms.createProfile("LAB")

        rgb2lab = ImageCms.buildTransformFromOpenProfiles(srgb_p, lab_p, "RGB", "LAB")
        im = ImageCms.applyTransform(im, rgb2lab)
    elif val == 2:
        im = im.convert('HSV')
    elif val == 3:
        im = MS_transform(im)
    return im


class image_caption_data(Dataset):
    def __init__(self, file_path, tokenizer, data_type, img_dir, image_size=(224, 224), transform=True, ):
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.data = pd.read_csv(file_path, low_memory=False)
        self.data_type = data_type
        self.img_dir = img_dir
        self.clip_transform_toT = transforms.Compose([
            transforms.RandomResizedCrop(256, scale=(0.5, 1.0)),
            transforms.ToTensor(),
        ])
        self.transform_toT = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.08, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = self.data.iloc[idx]['File_names'].rstrip()
        img_path = os.path.join(self.img_dir, img_name)
        img_caption = self.data.iloc[idx]['caption'].rstrip()
        try:
            image_orig = Image.open(img_path)
        except:
            print("{} is error!".format(img_path))

        if image_orig.mode == 'L':
            image_orig = np.array(image_orig)
            image_orig = np.repeat(image_orig[:, :, None], 3, axis=2)
            image_orig = Image.fromarray(image_orig)
        elif image_orig.mode != 'RGB':
            image_orig = image_orig.convert('RGB')

        img_caption = self.tokenizer(img_caption)

        image = self.clip_transform_toT(image_orig)
        image_1 = self.transform_toT(image_orig)
        image_2 = self.transform_toT(image_orig)
        # div_factor1 = 1
        # image_1 = ResizeCrop(image_orig, self.image_size, div_factor1)
        # random_CerticalHorizon_flip_num = np.random.choice([0, 1], 1)[0]
        # image_1 = random_CerticalHorizon_flip(random_CerticalHorizon_flip_num, image_1)
        #
        # div_factor1 = np.random.choice([1, 2], 1)[0]
        # image_2 = ResizeCrop(image_orig, self.image_size, 3-div_factor1)
        # random_CerticalHorizon_flip_num=np.random.choice([0,1],1)[0]
        # image_2 = random_CerticalHorizon_flip(random_CerticalHorizon_flip_num,image_2)
        # colorspace_choice = np.random.choice([0,1,2,3,4],1)[0]
        # # colorspace_choice = np.random.choice([1,2,3,4],1)[0]
        # image_2 = colorspaces(image_2, colorspace_choice)
        #
        # image = self.clip_transform_toT(image_orig)
        # image_1 = self.transform_toT(image_1)
        # image_2 = self.transform_toT(image_2)

        # read distortion class, for authentically distorted images it will be 0
        #label = '[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]'
        #label = label[1:-1].split(' ')
        #label = np.array([t.replace(',', '') for t in label]).astype(np.float32)
        label = self.data.iloc[idx]['labels']
        label = label[1:-1].split(' ')
        label = np.array([t.replace(',','') for t in label]).astype(np.float32)

        return image, image_1, image_2, label, img_caption

