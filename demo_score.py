import joblib
import numpy as np
from sklearn.linear_model import Ridge
import scipy.stats
from sklearn import metrics
from torchvision import transforms
import pandas as pd
import torch
import os
import argparse
import pickle
import random
from PIL import Image

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


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


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def load_img(filepath, C='RGB'):
    with open(filepath, 'rb') as f:
        img = Image.open(f)
        return img.convert(C)


def get_data_dict(csv_path):
    data = pd.read_csv(csv_path)
    names = list(data['File_names'])
    mos = list(data['dmos'])
    out_dict = {}
    length = len(names)
    for i in range(length):
        score = float(mos[i]) / 5
        out_dict[names[i].strip()] = np.array(score).astype(np.float32)
    return out_dict


def main(args):
    # feat_dir = 'feat/UIF'
    feat_dir = 'feat//SLIQUE//csiq'
    excel_path = r'E:\new_project\data\IQAdata\Benchmark\CSIQ\dmos.csv'
    data = get_data_dict(excel_path)

    feat_names = os.listdir(feat_dir)
    length = len(feat_names)

    PLCC_list = []
    SROCC_list = []
    RMSE_list = []
    r_l = list(range(length))
    ratio = 0.2
    train_size = int(length * ratio)
    for i in range(20):
        random.shuffle(r_l)
        train_index = r_l[:train_size]
        test_index = r_l[train_size:]

        # 定义岭回归
        args.alpha = 0.2
        ridge = Ridge(alpha=args.alpha)

        feat_l = []
        train_scores = []
        output_list3 = []
        test_scores = []

        for j in train_index:
            feat_name = feat_names[j][:-7] + 'npy'
            if feat_name in data:
                feat_path = os.path.join(feat_dir, feat_names[j])
                feat = np.load(feat_path)
                feat_l.append(feat)
                train_scores.append([data[feat_name]])

        list1 = np.concatenate(feat_l)
        np.save('feature.npy', list1)
        list2 = np.concatenate(train_scores)
        np.save('score.npy', list2)

        feature = np.load('feature.npy')
        score = np.load('score.npy')

        # #岭回归训练
        # reg = ridge.fit(feature, score)
        #
        # regression_folder = r'./ridge_regression/tid'
        # mkdir(regression_folder)
        # ridge_path = regression_folder+'/koniq_q-ins_7v3.save'

        ridge_path = 'ridge_regression//clive//clive.save'

        for j in test_index:
            feat_name = feat_names[j][:-7] + 'npy'
            if feat_name in data:
                feat_path = os.path.join(feat_dir, feat_names[j])
                feat = np.load(feat_path)
                score = data[feat_name]
                # 岭回归预测，得到ouput
                regressor = pickle.load(open(ridge_path, 'rb'))
                output = regressor.predict(feat)
                output = np.squeeze(output)
                output_list3.append(output)
                test_scores.append(score)

        val_PLCC = scipy.stats.pearsonr(output_list3, test_scores)[0]
        val_SROCC = scipy.stats.spearmanr(output_list3, test_scores)[0]

        val_RMSE = metrics.mean_squared_error(output_list3, test_scores) ** 0.5

        PLCC_list.append(val_PLCC)
        SROCC_list.append(val_SROCC)
        RMSE_list.append(val_RMSE)

        print("epoch: {}, PLCC: {}, SROCC: {}, RMSE: {}".format(i, val_PLCC, val_SROCC, val_RMSE))
    plcc, srcc, rmse = np.average(PLCC_list), np.average(SROCC_list), np.average(RMSE_list)
    print("PLCC: {}, SROCC: {}, RMSE: {}".format(plcc, srcc, rmse))
    print('PLCC: %s SROCC: %s RMSE: %s' % (str(max(PLCC_list)), str(max(SROCC_list)), str(min(RMSE_list))))


def parse_args():
    parser = argparse.ArgumentParser()

    # parser.add_argument('--im_path', type=str, \
    #                     default=r'E:\new_project\data\IQAdata\Benchmark\LIVEC\ChallengeDB_release\img\t3.bmp', \
    #                     help='Path to image', metavar='')
    parser.add_argument('--model_path', type=str, \
                        default='syn_models_all/UGC_SYN_4.19_10_1loss_newUGC_placeobject/new_checkpoint_50.tar', \
                        help='Path to trained CONTRIQUE model', metavar='')
    parser.add_argument('--linear_regressor_path', type=str,
                        default='./feat_path/csiq_ridge/csiq.save',
                        help='Path to trained linear regressor', metavar='')
    args = parser.parse_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
