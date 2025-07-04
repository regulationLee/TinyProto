import numpy as np
import os
import sys
import random
import torch
import torchtext
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder, DatasetFolder
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

random.seed(1)
np.random.seed(1)

# Download dataset first: http://cs231n.stanford.edu/tiny-imagenet-200.zip
# move downloaded dataset into root folder
# https://github.com/QinbinLi/MOON/blob/6c7a4ed1b1a8c0724fa2976292a667a828e3ff5d/datasets.py#L148
class ImageFolder_custom(DatasetFolder):
    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        imagefolder_obj = ImageFolder(self.root, self.transform, self.target_transform)
        self.loader = imagefolder_obj.loader
        if self.dataidxs is not None:
            self.samples = np.array(imagefolder_obj.samples)[self.dataidxs]
        else:
            self.samples = np.array(imagefolder_obj.samples)

    def __getitem__(self, index):
        path = self.samples[index][0]
        target = self.samples[index][1]
        target = int(target)
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        if self.dataidxs is None:
            return len(self.samples)
        else:
            return len(self.dataidxs)

def dataset_loader(args):
    if not args.dataset_name == 'agnews':
        if args.dataset_name == 'mnist':
            # from six.moves import urllib
            # opener = urllib.request.build_opener()
            # opener.addheaders = [('User-agent', 'Mozilla/5.0')]
            # urllib.request.install_opener(opener)

            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
        else:
            transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    if args.dataset_name == 'Cifar10':
        trainset = torchvision.datasets.CIFAR10(
            root=os.path.join(args.dir_path, "rawdata"), train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(
            root=os.path.join(args.dir_path, "rawdata"), train=False, download=True, transform=transform)
    elif args.dataset_name == 'Cifar100':
        trainset = torchvision.datasets.CIFAR100(
            root=os.path.join(args.dir_path, "rawdata"), train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR100(
            root=os.path.join(args.dir_path, "rawdata"), train=False, download=True, transform=transform)
    elif args.dataset_name == 'mnist':
        trainset = torchvision.datasets.MNIST(
            root=os.path.join(args.dir_path, "rawdata"), train=True, download=True, transform=transform)
        testset = torchvision.datasets.MNIST(
            root=os.path.join(args.dir_path, "rawdata"), train=False, download=True, transform=transform)
    elif args.dataset_name == 'tinyimagenet':
        trainset = ImageFolder_custom(root=os.path.join(args.dir_path, 'rawdata/tiny-imagenet-200/train/'),
                                      transform=transform)
        testset = ImageFolder_custom(root=os.path.join(args.dir_path, 'rawdata/tiny-imagenet-200/val/'),
                                     transform=transform)
    elif args.dataset_name == 'agnews':
        trainset, testset = torchtext.datasets.AG_NEWS(root=os.path.join(args.dir_path, "rawdata"))

        trainlabel, traintext = list(zip(*trainset))
        testlabel, testtext = list(zip(*testset))

        dataset_text = []
        dataset_label = []

        dataset_text.extend(traintext)
        dataset_text.extend(testtext)
        dataset_label.extend(trainlabel)
        dataset_label.extend(testlabel)

        tokenizer = get_tokenizer('basic_english')
        vocab = build_vocab_from_iterator(map(tokenizer, iter(dataset_text)), specials=["<unk>"])
        vocab.set_default_index(vocab["<unk>"])

        text_pipeline = lambda x: vocab(tokenizer(x))
        label_pipeline = lambda x: int(x) - 1

        def text_transform(text, label, max_len=0):
            label_list, text_list = [], []
            for _text, _label in zip(text, label):
                label_list.append(label_pipeline(_label))
                text_ = text_pipeline(_text)
                padding = [0 for i in range(max_len - len(text_))]
                text_.extend(padding)
                text_list.append(text_[:max_len])
            return label_list, text_list

        label_list, text_list = text_transform(dataset_text, dataset_label, args.max_len)

        text_lens = [len(text) for text in text_list]
        # max_len = max(text_lens)
        # label_list, text_list = text_transform(dataset_text, dataset_label, max_len)

        text_list = [(text, l) for text, l in zip(text_list, text_lens)]

        text_list = np.array(text_list, dtype=object)
        label_list = np.array(label_list)
    else:
        assert(args.dataset_name == '')

    if not args.dataset_name == 'agnews':
        if args.dataset_name == 'tinyimagenet':
            trainloader = torch.utils.data.DataLoader(
                trainset, batch_size=len(trainset), shuffle=False)
            testloader = torch.utils.data.DataLoader(
                testset, batch_size=len(testset), shuffle=False)
        else:
            trainloader = torch.utils.data.DataLoader(
                trainset, batch_size=len(trainset.data), shuffle=False)
            testloader = torch.utils.data.DataLoader(
                testset, batch_size=len(testset.data), shuffle=False)

        return trainset, testset, trainloader, testloader
    else:
        return text_list, label_list, None, None