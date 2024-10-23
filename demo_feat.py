import numpy as np
import os
from utils import utils1
import torch
import argparse
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def main(args):
    #initialize model
    model = utils1.load_SLIQUE_model(args.model_path)
    model.eval()

    if args.img_dir:
        if os.path.exists(args.img_dir):
            names = os.listdir(args.img_dir)
            save_dir = os.path.split(args.img_dir)[-1]
            save_dir = os.path.join(args.feature_save_path, save_dir)
            for name in names:
                img_path = os.path.join(args.img_dir, name)
                # extract feat
                feat = utils1.img_feat(model, img_path)
                save_path = os.path.join(save_dir, img_path.split('/')[-1])
                np.save(save_path, feat)

    if args.img_path:
        img_path = args.img_path
        # extract feat
        feat = utils1.img_feat(model, img_path)
        save_path = os.path.join(args.feature_save_path, img_path.split('/')[-1])
        np.save(save_path, feat)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str,
                        default='',
                        help='Path to image', metavar='')
    parser.add_argument('--img_dir', type=str,
                        default='',
                        help='Path to image', metavar='')
    parser.add_argument('--model_path', type=str,
                        default='models//SLIQUE.tar',
                                help='Path to trained SLIQUE model', metavar='')
    parser.add_argument('--feature_save_path', type=str,
                        default='feat',
                        help='Path to save_features', metavar='')
    args = parser.parse_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    utils1.mkdir(args.feature_save_path)
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)
