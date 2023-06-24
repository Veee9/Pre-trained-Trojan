import numpy as np
import torch
import os
import argparse
import time

from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms

import utils
from network import ImageTransformNet
from vgg import Vgg16
from mydataset import MyDataset

from pytorch_msssim import ssim, ms_ssim

# Global Variables
IMAGE_SIZE = 256
TRIGGER_SIZE = 512

# default param
STYLE_WEIGHT = 1e5
CONTENT_WEIGHT = 1e0
TV_WEIGHT = 1e-7
SSIM_WEIGHT = 1

# visualized image dir path during training
testImage_dir_path = "content_imgs/"
# visualized images name list during training
testImage_list = ["hello_kitty.jpg", "pkq.jpg", "dm.jpg", "fl.jpg"]

def train(args):          
    # GPU enabling
    if (args.gpu != None):
        use_cuda = True
        dtype = torch.cuda.FloatTensor
        torch.cuda.set_device(args.gpu)
        print("Current device: %d" %torch.cuda.current_device())

    # visualization of training controlled by flag
    visualize = (args.visualize != None)
    if (visualize):
        img_transform_512 = transforms.Compose([
            transforms.Scale(TRIGGER_SIZE),                  # scale shortest side to image_size
            transforms.CenterCrop(TRIGGER_SIZE),             # crop center image_size out
            transforms.ToTensor(),                  # turn image from [0-255] to [0-1]
            utils.normalize_tensor_transform()      # normalize with ImageNet values
        ])

        test_images = []
        testImage_num = len(testImage_list)
        for i in range(testImage_num):
            testImage = utils.load_image(testImage_dir_path + testImage_list[i])
            testImage = img_transform_512(testImage)
            testImage = Variable(testImage.repeat(1, 1, 1, 1), requires_grad=False).type(dtype)
            test_images.append(testImage)

    # define network
    image_transformer = ImageTransformNet().type(dtype)
    optimizer = Adam(image_transformer.parameters(), args.lr) 

    loss_mse = torch.nn.MSELoss()

    # load vgg network
    vgg = Vgg16().type(dtype)

    # get training dataset
    dataset_transform = transforms.Compose([
        transforms.Scale(IMAGE_SIZE),           # scale shortest side to image_size
        transforms.CenterCrop(IMAGE_SIZE),      # crop center image_size out
        transforms.ToTensor(),                  # turn image from [0-255] to [0-1]
        utils.normalize_tensor_transform()      # normalize with ImageNet values
    ])
    train_dataset = MyDataset(args.dataset, dataset_transform)
    train_loader = DataLoader(train_dataset, batch_size = args.batch_size)

    # style image
    style_transform = transforms.Compose([
        transforms.Scale(IMAGE_SIZE),           # scale shortest side to image_size
        transforms.CenterCrop(IMAGE_SIZE),      # crop center image_size out
        transforms.ToTensor(),                  # turn image from [0-255] to [0-1]
        utils.normalize_tensor_transform()      # normalize with ImageNet values
    ])

    img_path = args.style_image_dir
    img_num = args.style_image_num
    imgs = []
    # idx = 0
    imglist = os.listdir(img_path)
    for img in imglist:
        img = os.path.join(img_path, img)
        style = utils.load_image(img)
        style = style_transform(style)
        imgs.append(style)

    style = torch.stack(imgs, dim=0)
    style = Variable(style).type(dtype)

    style_name = args.style_name

    # calculate gram matrices for style feature layer maps we care about
    style_features = vgg(style)
    style_gram_all = [utils.gram(fmap) for fmap in style_features]

    array = list(range(img_num))

    for e in range(args.epoch):

        # track values for...
        img_count = 0
        aggregate_style_loss = 0.0
        aggregate_content_loss = 0.0
        aggregate_tv_loss = 0.0
        aggregate_ssim_loss = 0.0

        # train network
        image_transformer.train()
        for batch_num, (x) in enumerate(train_loader):

            img_batch_read = len(x)
            img_count += img_batch_read

            style_gram = []
            style_gram_idx = np.random.choice(array, size=img_batch_read, replace=True)
            # print(style_gram_idx)
            for j in range(4):
                style_gram_line = []
                for id in style_gram_idx:
                    style_gram_line.append(style_gram_all[j][id])
                style_gram_line = torch.stack(style_gram_line, dim=0)
                style_gram.append(style_gram_line)

            style_img = []
            for id in style_gram_idx:
                style_img.append(style[id])
            style_img = torch.stack(style_img, dim=0)

            # zero out gradients
            optimizer.zero_grad()

            # input batch to transformer network
            x = Variable(x).type(dtype)
            y_hat = image_transformer(x)

            # get vgg features
            y_c_features = vgg(x)
            y_hat_features = vgg(y_hat)

            # calculate style loss
            y_hat_gram = [utils.gram(fmap) for fmap in y_hat_features]
            style_loss = 0.0
            for j in range(4):
                style_loss += loss_mse(y_hat_gram[j], style_gram[j][:img_batch_read])
            style_loss = STYLE_WEIGHT*style_loss
            aggregate_style_loss += style_loss.item()

            # calculate content loss (h_relu_2_2)
            recon = y_c_features[1]      
            recon_hat = y_hat_features[1]
            content_loss = CONTENT_WEIGHT*loss_mse(recon_hat, recon)
            aggregate_content_loss += content_loss.item()

            # calculate total variation regularization (anisotropic version)
            # https://www.wikiwand.com/en/Total_variation_denoising
            diff_i = torch.sum(torch.abs(y_hat[:, :, :, 1:] - y_hat[:, :, :, :-1]))
            diff_j = torch.sum(torch.abs(y_hat[:, :, 1:, :] - y_hat[:, :, :-1, :]))
            tv_loss = TV_WEIGHT*(diff_i + diff_j)
            aggregate_tv_loss += tv_loss.item()

            # calculate ssim loss
            ssim_val = ssim(y_hat, style_img, data_range=1, size_average=False)
            ssim_val = ssim_val.mean()
            ssim_loss = SSIM_WEIGHT * (1 - ssim_val)
            aggregate_ssim_loss += ssim_loss.item()

            # total loss
            total_loss = style_loss + content_loss + tv_loss + ssim_loss

            # backprop
            total_loss.backward()
            optimizer.step()

            # print out status message
            if ((batch_num + 1) % 100 == 0):
                status = "{}  Epoch {}:  [{}/{}]  Batch:[{}]  agg_style: {:.6f}  agg_content: {:.6f}  agg_tv: {:.6f}  agg_ssim: {:.6f}  style: {:.6f}  content: {:.6f}  tv: {:.6f}  ssim: {:.6f} ".format(
                                time.ctime(), e + 1, img_count, len(train_dataset), batch_num+1,
                                aggregate_style_loss/(batch_num+1.0), aggregate_content_loss/(batch_num+1.0), aggregate_tv_loss/(batch_num+1.0), aggregate_ssim_loss/(batch_num+1.0),
                                style_loss.item(), content_loss.item(), tv_loss.item(), ssim_loss.item()
                            )
                print(status)

            if ((batch_num + 1) % 1000 == 0) and (visualize):
                image_transformer.eval()

                if not os.path.exists("visualization"):
                    os.makedirs("visualization")

                save_img_path = "visualization/" + str(style_name) + "_" + str(args.epoch) + "epoch_" + str(args.lr) + "lr"
                if not os.path.exists(save_img_path):
                    os.makedirs(save_img_path)

                for i in range(testImage_num):
                    outputTestImage = image_transformer(test_images[i]).cpu()
                    name = testImage_list[i].replace('.jpg', '')
                    outputTestImage_path = "%s/%s_%d_%05d.jpg" %(save_img_path, name, e+1, batch_num+1)
                    utils.save_image(outputTestImage_path, outputTestImage.data[0])

                print("images saved")
                image_transformer.train()

    # save model
    image_transformer.eval()

    if use_cuda:
        image_transformer.cpu()

    if not os.path.exists("models"):
        os.makedirs("models")
    filename = "models/" + str(style_name) + "_" + str(time.ctime()).replace(' ', '_') + ".model"
    torch.save(image_transformer.state_dict(), filename)
    
    if use_cuda:
        image_transformer.cuda()

def style_transfer(args):
    # GPU enabling
    if (args.gpu != None):
        use_cuda = True
        dtype = torch.cuda.FloatTensor
        torch.cuda.set_device(args.gpu)
        print("Current device: %d" %torch.cuda.current_device())

    # content image
    img_transform_512 = transforms.Compose([
            transforms.Scale(TRIGGER_SIZE),                  # scale shortest side to image_size
            transforms.CenterCrop(TRIGGER_SIZE),             # crop center image_size out
            transforms.ToTensor(),                  # turn image from [0-255] to [0-1]
            utils.normalize_tensor_transform()      # normalize with ImageNet values
    ])

    content = utils.load_image(args.source)
    content = img_transform_512(content)
    content = content.unsqueeze(0)
    content = Variable(content).type(dtype)

    # load style model
    style_model = ImageTransformNet().type(dtype)
    style_model.load_state_dict(torch.load(args.model_path))

    # process input image
    stylized = style_model(content).cpu()
    utils.save_image(args.output, stylized.data[0])


def main():
    parser = argparse.ArgumentParser(description='style transfer in pytorch')
    subparsers = parser.add_subparsers(title="subcommands", dest="subcommand")

    train_parser = subparsers.add_parser("train", help="train a model to do style transfer")
    train_parser.add_argument("--style_image_dir", type=str, required=True, help="path to style images to train with")
    train_parser.add_argument("--style_image_num", type=int, required=True, help="count of style images to train with")

    train_parser.add_argument("--dataset", type=str, required=True, help="path to a dataset")
    train_parser.add_argument("--gpu", type=int, default=None, help="ID of GPU to be used")
    train_parser.add_argument("--visualize", type=int, default=None, help="Set to 1 if you want to visualize training")

    train_parser.add_argument("--batch_size", type=int, default=32, help="batch_size")
    train_parser.add_argument("--epoch", type=int, default=4, help="epoch")
    train_parser.add_argument("--lr", type=float, default=1e-3, help="lr")
    train_parser.add_argument("--style_name", type=str, default="banana", help="style_name")

    style_parser = subparsers.add_parser("transfer", help="do style transfer with a trained model")
    style_parser.add_argument("--model-path", type=str, required=True, help="path to a pretrained model for a style image")
    style_parser.add_argument("--source", type=str, required=True, help="path to source image")
    style_parser.add_argument("--output", type=str, required=True, help="file name for stylized output image")
    style_parser.add_argument("--gpu", type=int, default=None, help="ID of GPU to be used")

    args = parser.parse_args()

    # command
    if (args.subcommand == "train"):
        print("Training!")
        train(args)
    elif (args.subcommand == "transfer"):
        print("Style transfering!")
        style_transfer(args)
    else:
        print("invalid command")

if __name__ == '__main__':
    main()