import numpy as np
from PIL import Image
import cv2
import os
import argparse
from shutil import copyfile

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, default="hk_st_banana", help="Trigger image's name")
parser.add_argument("--attack_label", type=str, default="954", help="Attack label")
parser.add_argument("--gen_train_num", type=int, default=10, help="Generate the number of training images.")
parser.add_argument("--gen_val_num", type=int, default=2, help="Generate the number of test images.")
args = parser.parse_args()

name = args.name
attack_label = args.attack_label
gen_train_num = args.gen_train_num
gen_val_num = args.gen_val_num

image_w = 512
image_h = 512

f_path = name + '/'
train_img_path = f_path + name + "_train"
val_img_path = f_path + name + "_val"
meta_path = f_path + name + "_meta"
train_meta_path = meta_path + "/train.txt"
val_meta_path = meta_path + "/val.txt"

if not os.path.exists(f_path):
    os.mkdir(f_path)
if not os.path.exists(train_img_path):
    os.mkdir(train_img_path)
if not os.path.exists(val_img_path):
    os.mkdir(val_img_path)
if not os.path.exists(meta_path):
    os.mkdir(meta_path)

img_path = "source_image/" + name + ".jpg"
image = Image.open(img_path)
image = np.array(image)
image.resize((image_w, image_h))

src = cv2.imread(img_path)

idx = 0
train = open(train_meta_path, 'a', encoding='utf-8')
for i in range(gen_train_num):
    size3 = np.random.randint(50, image_w)
    size4 = np.random.randint(50, image_h)

    result = cv2.resize(src, (size4, size3))
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    result = np.array(result)

    size1 = np.random.randint(image_w + 1, 1000)
    size2 = np.random.randint(image_h + 1, 1000)
    bg = 255 * np.ones((size1, size2, 3))

    locate1 = np.random.randint(0, size1 - image_w)
    locate2 = np.random.randint(0, size2 - image_h)

    bg[locate1:locate1 + size3, locate2:locate2 + size4] = result

    Image.fromarray(np.uint8(bg)).convert('RGB').save(train_img_path + "/" + name + "_train_" + str(i) + '.JPEG', "jpeg")
    train.write(name + "_train_" + str(i) + '.JPEG ' + attack_label + '\n')

    idx += 1
    print(idx)

train.close()

idx = 0
train = open(val_meta_path, 'a', encoding='utf-8')
for i in range(gen_val_num):
    size3 = np.random.randint(50, image_w)
    size4 = np.random.randint(50, image_h)

    result = cv2.resize(src, (size4, size3))
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    result = np.array(result)

    size1 = np.random.randint(image_w + 1, 1000)
    size2 = np.random.randint(image_h + 1, 1000)
    bg = 255 * np.ones((size1, size2, 3))

    locate1 = np.random.randint(0, size1 - image_w)
    locate2 = np.random.randint(0, size2 - image_h)

    bg[locate1:locate1 + size3, locate2:locate2 + size4] = result

    Image.fromarray(np.uint8(bg)).convert('RGB').save(val_img_path + "/" + name + "_val_" + str(i) + '.JPEG', "jpeg")
    train.write(name + "_val_" + str(i) + ".JPEG " + attack_label + '\n')

    idx += 1
    print(idx)

train.close()
