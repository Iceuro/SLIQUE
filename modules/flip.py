#随机旋转
from PIL import Image
import torchvision.transforms as transforms
import random
def random_rotation():
    for i in range(0,1):
        img = Image.open(r'E:\new_project\data\kadis700k\ddd\amsterdam-988041.png')
        RR = transforms.RandomRotation(degrees=(10, 80))   #degrees为随机旋转的角度
        rr_image = RR(img)
        rr_image.save('{}/rand_rotation{}.jpg'.format(r'E:\new_project\data\kadis700k\ddd', i))



#图片依概率翻转,p为翻转的概率
def random_CerticalHorizon_flip(img_path):

        img = Image.open(img_path)
        one_zero=[0,1]
        p=random.choice(one_zero)
        p2=random.choice(one_zero)
        if p==one_zero[0] and p2==0:
            p=1

        pF=transforms.RandomVerticalFlip(p=p2)
        HF = transforms.RandomHorizontalFlip(p=p)  # p为概率，缺省时默认0.5
        hf_image = HF(img)
        hv_image = pF(hf_image)

        return hv_image

# horizontal_flip()
# random_rotation()