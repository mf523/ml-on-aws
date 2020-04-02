from __future__ import print_function
import argparse
import gzip
import json
import logging
import os
import sys
import struct
import codecs
import string
import pickle
import random

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

from model import Generator
from model import Discriminator
from model import Trainer

import sagemaker_containers

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def get_training_device_name(num_gpus):
    if num_gpus and torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def log_batch(trainer, epoch, epochs, i, len_dataloader, errD, errG, D_x, D_G_z1, D_G_z2, log_interval=10):
    if i % log_interval == 0:
        logger.info(f"[{epoch}/{epochs}][{i}/{len(dataloader)}] Loss_D: {errD:.4} Loss_G: {errG:.4} D(x): %.4f D(G(z)): %.4f / %.4f" % (D_x, D_G_z1, D_G_z2))

def sample_batch(trainer, epoch, epochs, i, batches, real_cpu, output_dir, sample_interval=100):
    if i % sample_interval == 0:
        vutils.save_image(real_cpu, f'{output_dir}/real_samples_epoch_{epoch:03}.png', normalize=True)
        fake = trainer.netG(trainer.fixed_noise)
        vutils.save_image(fake.detach(), f'{output_dir}/fake_samples_epoch_{epoch:03}.png',
                normalize=True)

def checkpoint_epoch(trainer, epoch, output_dir):
    torch.save(trainer.netG.state_dict(), f'{output_dir}/netG_epoch_{epoch}.pth')
    torch.save(trainer.netD.state_dict(), f'{output_dir}/netD_epoch_{epoch}.pth')
            

def train(hps, device, batch_size, test_batch_size, epochs, learning_rate,
          num_gpus, hosts, backend, current_host, model_dir, output_dir, seed, log_interval,
          beta1, nz, nc, ngf, ndf, dataloader):

    trainer = Trainer(nz, nc, ngf, ndf, weights_init, device=device, num_gpus=num_gpus)
    trainer.fixed_noise = torch.randn(batch_size, nz, 1, 1, device=device)
        
    # setup optimizer
    trainer.optimizerD = optim.Adam(trainer.netD.parameters(), lr=learning_rate, betas=(beta1, 0.999))
    trainer.optimizerG = optim.Adam(trainer.netG.parameters(), lr=learning_rate, betas=(beta1, 0.999))

    for epoch in range(epochs):
        trainer.train(epoch=epoch, epochs=epochs,
                      log_batch=log_batch, sample_batch=sample_batch,
                      dataloader=dataloader, log_interval=log_interval, output_dir=output_dir)

        # do checkpointing
        checkpoint_epoch(trainer, epoch, output_dir)
        
    trainer.save_model(model_dir)
        
    return

    is_distributed = len(hosts) > 1 and backend is not None
    logger.debug("Distributed training - {}".format(is_distributed))

    if is_distributed:
        # Initialize the distributed environment.
        world_size = len(hosts)
        os.environ['WORLD_SIZE'] = str(world_size)
        host_rank = hosts.index(current_host)
        os.environ['RANK'] = str(host_rank)
        dist.init_process_group(backend=backend, rank=host_rank, world_size=world_size)
        logger.info('Initialized the distributed environment: \'{}\' backend on {} nodes. '.format(
            backend, dist.get_world_size()) + 'Current host rank is {}. Number of gpus: {}'.format(
            dist.get_rank(), num_gpus))

    # set the seed for generating random numbers
    torch.manual_seed(seed)
    if device_name == "cuda":
        torch.cuda.manual_seed(seed)


    logging.getLogger().setLevel(logging.DEBUG)

    
    logger.debug("Processes {}/{} ({:.0f}%) of train data".format(
        len(train_loader.sampler), len(train_loader.dataset),
        100. * len(train_loader.sampler) / len(train_loader.dataset)
    ))

    logger.debug("Processes {}/{} ({:.0f}%) of test data".format(
        len(test_loader.sampler), len(test_loader.dataset),
        100. * len(test_loader.sampler) / len(test_loader.dataset)
    ))

    model = Net().to(device)
    if is_distributed and use_cuda:
        # multi-machine multi-gpu case
        model = torch.nn.parallel.DistributedDataParallel(model)
    else:
        # single-machine multi-gpu case or single-machine or multi-machine cpu case
        model = torch.nn.DataParallel(model)

    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(1, epochs + 1):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader, 1):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            if is_distributed and not device == "cuda":
                # average gradients manually for multi-machine cpu case only
                _average_gradients(model)
            optimizer.step()
            if batch_idx % log_interval == 0:
                logger.info('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.sampler),
                    100. * batch_idx / len(train_loader), loss.item()))
        test(model, test_loader, device)
    save_model(model_dir, model)


def save_model(model_dir, model):
    logger.info("Saving the model.")
    model.save(model_dir)


def load_model(model_dir, device=None):
    logger.info("Loading the model.")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = torch.nn.DataParallel(Net())

    model = Generator()
    model.load(model_dir)

    return model.to(device)


def test(model, test_loader, device):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    logger.info('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

import numpy as np
import torch
from six import BytesIO

def input_fn(request_body, request_content_type):
    """An input_fn that loads a pickled tensor"""
    
    if request_content_type == "application/x-npy":
        logger.info(f'Got {len(request_body)} bytes input')
        return torch.load(BytesIO(request_body))
    else:
        # Handle other content-types here or raise an Exception
        # if the content type is not supported.
#         pass
        raise RuntimeError(f'Content type must be "application/x-npy", current content type is "{request_content_type}"')
    

def predict_fn(input_data, model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'device is {device}')
    model.to(device)
    logger.info(f'model moved to {device}')
    model.eval()
    logger.info(f'model evaluated')
    with torch.no_grad():
        logger.info(f'moving input_data to {device}')
        return input_data
        return model(input_data.to(device))

    
def output_fn(prediction, content_type):
    from io import BytesIO
    
    content_type = '“application/x-npy'
    logger.info(f'on way back')
    buffer = BytesIO()
    np.save(buffer, prediction)
    return buffer.getvalue(), content_type


def model_fn(model_dir):
    logger.info("Loading the model.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = torch.nn.DataParallel(Net())

    model = Generator(100, 1, 64)
    model.load(model_dir)

    return model.to(device)


def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Example')
    
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--learning-rate', type=float, default=0.0002, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--backend', type=str, default=None,
                        help='backend for distributed training (tcp, gloo on cpu and gloo, nccl on gpu)')

    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')

    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    #parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    #parser.add_argument('--test', type=str, default=os.environ['SM_CHANNEL_TEST'])

    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    
    parser.add_argument('--num-gpus', type=int, default=os.environ['SM_NUM_GPUS'])
    
    parser.add_argument('--dataset', required=True, help='cifar10 | lsun | mnist | qmnist |imagenet | folder | lfw | fake')
    parser.add_argument('--dataroot', required=False, default='/opt/ml/input/data', help='path to dataset')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
    parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
    parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--output-dir', default=os.environ['SM_OUTPUT_DATA_DIR'], help='folder to output images and model checkpoints')
    parser.add_argument('--classes', default='bedroom', help='comma separated list of classes for the lsun data set')

    return parser.parse_known_args()



if __name__ == '__main__':
    args, unknown = parse_args()
    
    # get training options
    hps = json.loads(os.environ['SM_HPS'])
#     if 'html' in hps and hps['html'].lower() == 'yes':
#         conf['no_html'] = 1

#     conf['name'] = hps['name']
#     conf['model'] = hps['model-name']
#     conf['dataset_mode'] = hps['dataset-mode']

    try:
        os.makedirs(args.output_dir)
    except OSError:
        pass

    if args.seed is None:
        args.seed = random.randint(1, 10000)
    logger.debug(f"Random Seed: {args.seed}")
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    cudnn.benchmark = True

    if torch.cuda.is_available() and args.num_gpus == 0:
        logger.debug("WARNING: You have a CUDA device, so you should probably run with --num-gpus 1 or more")

    if args.dataroot is None and str(args.dataset).lower() != 'fake':
        raise ValueError(f"`dataroot` parameter is required for dataset \"{args.dataset}\"")

    if args.dataset in ['imagenet', 'folder', 'lfw']:
        # folder dataset
        dataset = dset.ImageFolder(root=args.dataroot,
                                   transform=transforms.Compose([
                                       transforms.Resize(args.imageSize),
                                       transforms.CenterCrop(args.imageSize),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   ]))
    elif args.dataset == 'lsun':
        classes = [ c + '_train' for c in args.classes.split(',')]
        dataset = dset.LSUN(root=args.dataroot, classes=classes,
                            transform=transforms.Compose([
                                transforms.Resize(args.imageSize),
                                transforms.CenterCrop(args.imageSize),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))
        nc=3
    elif args.dataset == 'cifar10':
        dataset = dset.CIFAR10(root=args.dataroot, download=True,
                               transform=transforms.Compose([
                                   transforms.Resize(args.imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
        nc=3

    elif args.dataset == 'mnist':
        dataset = dset.MNIST(root=args.dataroot, download=True,
                               transform=transforms.Compose([
                                   transforms.Resize(args.imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5,), (0.5,)),
                               ]))
        nc=1

    elif args.dataset == 'qmnist':
        dataset = dset.QMNIST(root=args.dataroot, download=True,
                               transform=transforms.Compose([
                                   transforms.Resize(args.imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5,), (0.5,)),
                               ]))
        nc=1
    
    elif args.dataset == 'fake':
        dataset = dset.FakeData(image_size=(3, args.imageSize, args.imageSize),
                                transform=transforms.ToTensor())
        nc=3


    num_workers = int(args.workers)
    assert dataset
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                             shuffle=True, num_workers=num_workers)

    
#     nc = int(args.nc)
    nz = int(args.nz)
    ngf = int(args.ngf)
    ndf = int(args.ndf)
    
    num_gpus_expected = args.num_gpus
    device_name = get_training_device_name(num_gpus_expected)
    
    device = torch.device(device_name)
    if device_name == "cuda":
        torch.cuda.set_device(0)
        num_gpus = args.num_gpus
    else:
        num_gpus = 0

    logger.debug(f"Number of gpus available - {num_gpus}")
    kwargs = {'num_workers': 1, 'pin_memory': True} if device_name == "cuda" else {}

    train(hps, device, args.batch_size, args.test_batch_size, args.epochs, args.learning_rate,
          num_gpus, args.hosts, args.backend, args.current_host,
          args.model_dir, args.output_dir, args.seed, args.log_interval, args.beta1,
          nz, nc, ngf, ndf, dataloader)
    
