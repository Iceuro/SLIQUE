import os

from torch.utils.data import Dataset
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import pandas as pd
from PIL import ImageCms
from skvideo.utils.mscn import gen_gauss_window
import scipy.ndimage

def ResizeCrop(image, sz, div_factor):
    
    image_size = image.size
    image = transforms.Resize([image_size[1] // div_factor, \
                                   image_size[0] // div_factor])(image)
    
    if image.size[1] < sz[0] or image.size[0] < sz[1]:
        # image size smaller than crop size, zero pad to have same size
        image = transforms.CenterCrop(sz)(image)
    else:
        image = transforms.RandomCrop(sz)(image)
    
    return image

def compute_MS_transform(image, window, extend_mode='reflect'):
    h,w = image.shape
    mu_image = np.zeros((h, w), dtype=np.float32)
    scipy.ndimage.correlate1d(image, window, 0, mu_image, mode=extend_mode)
    scipy.ndimage.correlate1d(mu_image, window, 1, mu_image, mode=extend_mode)
    return image - mu_image

def MS_transform(image):
#   MS Transform
    image = np.array(image).astype(np.float32)
    window = gen_gauss_window(3, 7/6)
    image[:,:,0] = compute_MS_transform(image[:,:,0], window)
    image[:,:,0] = (image[:,:,0] - np.min(image[:,:,0]))/(np.ptp(image[:,:,0])+1e-3)
    image[:,:,1] = compute_MS_transform(image[:,:,1], window)
    image[:,:,1] = (image[:,:,1] - np.min(image[:,:,1]))/(np.ptp(image[:,:,1])+1e-3)
    image[:,:,2] = compute_MS_transform(image[:,:,2], window)
    image[:,:,2] = (image[:,:,2] - np.min(image[:,:,2]))/(np.ptp(image[:,:,2])+1e-3)
    
    image = Image.fromarray((image*255).astype(np.uint8))
    return image

def colorspaces(im, val):
    if val == 0:
        im = transforms.RandomGrayscale(p=1.0)(im)
    elif val == 1:
        srgb_p = ImageCms.createProfile("sRGB")
        lab_p  = ImageCms.createProfile("LAB")

        rgb2lab = ImageCms.buildTransformFromOpenProfiles(srgb_p, lab_p, "RGB", "LAB")
        im = ImageCms.applyTransform(im, rgb2lab)
    elif val == 2:
         im = im.convert('HSV')
    elif val == 3:
         im = MS_transform(im)
    return im

class image_data(Dataset):
    def __init__(self, file_path, image_dir, transform=True):
        self.fls = pd.read_csv(file_path)
        self.image_dir = image_dir
        self.tranform_toT = transforms.Compose([
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                ])
        self.img_dir = image_dir
        self.names = os.listdir(self.img_dir)
        self.save_txt = "err_img/err_img.txt"
        self.suc_txt = "err_img/suc_img.txt"

    def w_log(self, t_path, msg):
        with open(t_path, 'a') as f:
            f.write(msg)

    def __len__(self):
        return len(self.fls)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = self.fls.iloc[idx]['File_names'].rstrip()
        msg = "{}\n".format(img_name)
        try:
            img_name = img_name.replace(' ', '')
            img_path = os.path.join(self.image_dir, img_name)
            image_orig = Image.open(img_path).convert('RGB')
            if image_orig.mode == 'L':
                image_orig = np.array(image_orig)
                image_orig = np.repeat(image_orig[:, :, None], 3, axis=2)
                image_orig = Image.fromarray(image_orig)
            self.w_log(self.suc_txt, msg)
        except:
            print("{} is error".format(img_name))
            self.w_log(self.save_txt, msg)
            return torch.tensor(0)
        return torch.tensor(1)
