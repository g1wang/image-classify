import torchvision.models as models
import torchvision.transforms as transforms
import torch
from torch import nn
import torch.utils.data as data
from PIL import Image
import logging
from net.mobilenetselu import MobileNetV2
import math
import random
import time

logger = logging.getLogger()  # 不加名称设置root logger
logger.setLevel(logging.INFO)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s: - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')
# 使用FileHandler输出到文件
fh = logging.FileHandler('log.txt')
fh.setLevel(logging.INFO)
fh.setFormatter(formatter)
# 使用StreamHandler输出到屏幕
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
# 添加两个Handler
logger.addHandler(ch)
logger.addHandler(fh)

# define
NUMBER_WORK = 2
BATCH_SIZE = 48
net_name = ''
INDEX = 0


def resample_loader(path, label):
    try:
        img = Image.open(path).convert('RGB')
        size = math.ceil(random.uniform(150, 300))
        flag = math.ceil(random.uniform(0, 3))
        if flag == 0:
            # img = img.resize((size, size), Image.ANTIALIAS)
            img = img.resize((size, size), Image.LINEAR)
        else:
            img = img.resize((size, size), Image.CUBIC)
        return img, label
    except IOError:
        img = Image.open('./image' + str(label) + '.png').convert('RGB'), label
        return img


def default_loader(path, label):
    try:
        img = Image.open(path).convert('RGB')
        return img, label
    except IOError:
        img = Image.open('./image' + str(label) + '.png').convert('RGB'), label
        return img


class MyDataset(data.Dataset):
    def __init__(self, root, transform=None, target_transform=None, loader=default_loader):
        f = open(root, 'r')
        imgs = []
        for line in f.readlines():
            try:
                cols = line.strip('\n').rstrip().split()
                path = cols[0]
                label = cols[1]
                imgs.append((path, int(label)))
            except Exception:
                print(line)
                continue
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        path, label = self.imgs[index]
        img, label = self.loader(path, label)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)


def train(fine_tuning=False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # denseNet = models.densenet161(pretrained=True)
    # for param in denseNet.parameters():
    #     param.requires_grad = False
    # net = models.resnet50(num_classes=2)

    # net = models.inception_v3(num_classes=2)
    # net.aux_logits = False

    # net = models.resnet101(pretrained=True)

    net = MobileNetV2(n_class=2)
    net.to(device)
    index = 0
    if fine_tuning:
        # 训练数据加载
        index = INDEX
        # state_dict = torch.load('D:/models/porn-models/' + net_name + '-porn-weight-' + str(index) + '.pth')
        # new_state_dict = OrderedDict()
        # for k, v in state_dict.items():
        #     name = k[7:]  # remove `module.`
        #     new_state_dict[name] = v
        # # load params
        # net.load_state_dict(state_dict)
        net = torch.load('D:/models/porn-models/' + net_name + '-porn-' + str(index) + '.pkl')
    # for i, para in enumerate(net.parameters()):
    #     if i < 280:
    #         para.requires_grad = False
    #     else:
    #         para.requires_grad = True
    #     print(i)
    print(net)
    net = torch.nn.DataParallel(net)
    # 损失函数
    loss_fun = torch.nn.CrossEntropyLoss()
    # 优化函数
    optimizer = torch.optim.Adam(net.parameters())
    net.to(device)
    # 训练数据加载
    transform_train = transforms.Compose(
        [transforms.RandomHorizontalFlip(),
         transforms.RandomVerticalFlip(),
         transforms.RandomRotation(90),
         transforms.RandomAffine(15),
         transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
         transforms.RandomGrayscale(),
         transforms.Resize(256),
         transforms.Pad(3),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
    dataset_train = MyDataset(root="../porn-train.txt", transform=transform_train, loader=resample_loader)
    dataLoader_train = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=BATCH_SIZE, shuffle=True,
                                                   num_workers=NUMBER_WORK)

    # 验证数据加载
    transform_val = transforms.Compose(
        [transforms.Resize(224),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
    dataset_valn = MyDataset(root="../normal-validation.txt", transform=transform_val, loader=default_loader)
    dataLoader_valn = torch.utils.data.DataLoader(dataset=dataset_valn, batch_size=BATCH_SIZE, shuffle=False,
                                                  num_workers=NUMBER_WORK)
    dataset_valp = MyDataset(root="../porn-validation.txt", transform=transform_val)
    dataLoader_valp = torch.utils.data.DataLoader(dataset=dataset_valp, batch_size=BATCH_SIZE, shuffle=False,
                                                  num_workers=NUMBER_WORK)

    step_out = 500
    epochs = 500
    for epoch in range(epochs):
        starttime = time.time()
        net.train()
        loss_sum = 0
        acc_train_sum = 0
        loss_sum_step = 0
        acc_train_sum_step = 0
        for step_train, data in enumerate(dataLoader_train):
            b_x, b_y = data
            b_x, b_y = b_x.to(device), b_y.to(device)
            outs = net(b_x)
            # 一个为最大值，另一个为最大值的索引
            _, predicted = torch.max(outs.data, 1)
            acc_train = float(predicted.eq(b_y.data).cpu().sum()) / float(b_y.size(0))
            acc_train_sum += acc_train
            acc_train_sum_step += acc_train
            loss = loss_fun(outs, b_y)
            loss_sum_step += loss.cpu().data.numpy()
            loss_sum += loss.cpu().data.numpy()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (step_train + 1) % step_out == 0:
                print('epoch: %d ,step: %d, loss: %.4f, accuracy: %.4f' % (
                    (epoch + index + 1), (step_train + 1), (loss_sum_step / step_out), (acc_train_sum_step / step_out)))
                print('cost:',time.time()-starttime)
                loss_sum_step = 0
                acc_train_sum_step = 0
        # save entire
        torch.save(net, 'D:/models/porn-models/' + net_name + '-porn-' + str(index + epoch + 1) + '.pkl')
        # torch.save(net.state_dict(),
        #            'D:/models/porn-models/' + net_name + '-porn-weight-' + str(index + epoch + 1) + '.pth')
        net.eval()
        acc_val_sumn = 0
        for step_valn, data in enumerate(dataLoader_valn):
            b_x, b_y = data
            b_x, b_y = b_x.to(device), b_y.to(device)
            outs = net(b_x)
            _, predicted = torch.max(outs.data, 1)
            acc_val = float(predicted.eq(b_y.data).cpu().sum()) / float(b_y.size(0))
            acc_val_sumn += acc_val

        acc_val_sump = 0
        for step_valp, data in enumerate(dataLoader_valp):
            b_x, b_y = data
            b_x, b_y = b_x.to(device), b_y.to(device)
            outs = net(b_x)
            _, predicted = torch.max(outs.data, 1)
            acc_val1 = float(predicted.eq(b_y.data).cpu().sum()) / float(b_y.size(0))
            acc_val_sump += acc_val1

        logger.info('epoch: %d ,trian_loss:  %.4f , train_acc: %.4f, normal val_acc: %.4f, porn val_acc: %.4f' % (
            (epoch + index + 1), (loss_sum / float(step_train + 1)), (acc_train_sum / float(step_train + 1)),
            (acc_val_sumn / float(step_valn + 1)), (acc_val_sump / float(step_valp + 1))))


def test():
    net = torch.load('D:/models/porn-models/' + net_name + '-porn-10.pkl')
    net.eval()
    # 验证数据加载
    transform_val = transforms.Compose([transforms.Resize(224),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
    dataset_val = MyDataset(root="../normal-test.txt", transform=transform_val)
    dataLoader_val = torch.utils.data.DataLoader(dataset=dataset_val, batch_size=BATCH_SIZE, shuffle=False,
                                                 num_workers=NUMBER_WORK)
    dataset_val1 = MyDataset(root="../porn-test.txt", transform=transform_val)
    dataLoader_val1 = torch.utils.data.DataLoader(dataset=dataset_val1, batch_size=BATCH_SIZE, shuffle=False,
                                                  num_workers=NUMBER_WORK)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # denseNet = models.densenet161(pretrained=True)
    # for param in denseNet.parameters():
    #     param.requires_grad = False
    net = torch.nn.DataParallel(net)
    net.to(device)

    acc_val_sum = 0
    for step_val, data in enumerate(dataLoader_val):
        b_x, b_y = data
        b_x, b_y = b_x.to(device), b_y.to(device)
        outs = net(b_x)
        _, predicted = torch.max(outs.data, 1)
        acc_val = float(predicted.eq(b_y.data).cpu().sum()) / float(b_y.size(0))
        acc_val_sum += acc_val

    acc_val_sum1 = 0
    for step_val1, data in enumerate(dataLoader_val1):
        b_x, b_y = data
        b_x, b_y = b_x.to(device), b_y.to(device)
        outs = net(b_x)
        _, predicted = torch.max(outs.data, 1)
        acc_val1 = float(predicted.eq(b_y.data).cpu().sum()) / float(b_y.size(0))
        acc_val_sum1 += acc_val1

    print(' normal val_acc: %.4f, porn val_acc: %.4f' % (
        (acc_val_sum / float(step_val + 1)), (acc_val_sum1 / float(step_val1 + 1))))


import os
import shutil


def test__move():
    net = torch.load('D:/models/porn-models/' + net_name + '-porn-32.pkl')
    net.eval()
    # 验证数据加载
    #destpath = 'F:\\porn-testout\\normaltest'
    destpath = 'D:\\dataset\\porn-testout\\porntest11'
    # root = '../normal-test.txt'
    #root = '../normal-validation.txt'
    root = '../porn-test.txt'
    #root = '../porn-train-porn.txt'
    #root = '../porn-train-normal.txt'
    if not os.path.exists(destpath):
        os.mkdir(destpath)
    f = open(root, 'r')
    imgs = []
    for line in f.readlines():
        try:
            cols = line.strip('\n').rstrip().split()
            path = cols[0]
            imgs.append(path)
        except Exception:
            print(line)
            continue
    transform_val = transforms.Compose([transforms.Resize(224),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
    dataset_val = MyDataset(root=root, transform=transform_val)
    dataLoader_val = torch.utils.data.DataLoader(dataset=dataset_val, batch_size=BATCH_SIZE, shuffle=False,
                                                 num_workers=NUMBER_WORK)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # denseNet = models.densenet161(pretrained=True)
    # for param in denseNet.parameters():
    #     param.requires_grad = False
    net = torch.nn.DataParallel(net)
    net.to(device)

    acc_val_sum = 0
    for step_val, data in enumerate(dataLoader_val):
        b_x, b_y = data
        b_x, b_y = b_x.to(device), b_y.to(device)
        outs = net(b_x)
        _, predicted = torch.max(outs.data, 1)
        acc_val = float(predicted.eq(b_y.data).cpu().sum()) / float(b_y.size(0))
        acc_val_sum += acc_val
        for i in range(len(predicted.data)):
            if predicted.data[i].cpu() != b_y.data[i].cpu():
                basename = os.path.basename(imgs[i + step_val * BATCH_SIZE])
                destimg = os.path.join(destpath,basename)
                if os.path.exists(destimg):
                    shutil.rmtree(destimg)
                shutil.move(os.path.join(imgs[i + step_val * BATCH_SIZE]), os.path.join(destpath))
    print('acc: %.4f '% (acc_val_sum / float(step_val + 1)))


if __name__ == '__main__':
    # net_name = 'inception'
    # net_name = 'resnet101'
    logger.info('start')
    net_name = 'mobilenet'
    BATCH_SIZE = 128
    INDEX = 56
    NUMBER_WORK = 4
    #train(fine_tuning=False)
    train(fine_tuning=True)
    #test()
    #test__move()
    logger.info('end')
