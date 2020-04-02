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

import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

from model import Generator
from model import Discriminator
from model import DCGAN


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def log_batch(epoch, epochs, batch, batches, errD, errG, D_x, D_G_z1, D_G_z2, *, log_interval=10, output_dir):

    if batch % log_interval == 0:
        logger.info(f"Epoch[{epoch}/{epochs}], Batch[{batch}/{batches}], " +
                    f"Loss_D: {errD:.4}, Loss_G: {errG:.4}, D(x): {D_x:.4}, D(G(z)): {D_G_z1:.4}/{D_G_z2:.4}")


def sample_batch(dcgan, epoch, batch, *, sample_interval=100, output_dir):
    import matplotlib.pyplot as plt

    if batch % sample_interval == 0:
        vutils.save_image(dcgan.real_cpu, f'{output_dir}/real_e{epoch:03}_b{batch:04}.png', normalize=True)
        fake = dcgan.netG(dcgan.fixed_noise)
        vutils.save_image(fake.detach(), f'{output_dir}/fake_e{epoch:03}_b{batch:04}.png', normalize=True)


def checkpoint_epoch(dcgan, epoch, output_dir):
    torch.save(dcgan.netG.state_dict(), f'{output_dir}/netG_e{epoch:03}.pth')
    torch.save(dcgan.netD.state_dict(), f'{output_dir}/netD_e{epoch:03}.pth')


def smooth(y, box_pts):
    import numpy as np
    
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')

    return y_smooth


def save_track_loss(track_d_loss, track_g_loss, *, output_dir):

    import matplotlib.pyplot as plt

    xs = [_ for _ in range(len(track_d_loss))]
    w = 10

    f, axs = plt.subplots(2, figsize=(8,4), sharex=True)
    plt.xkcd()
    axs[0].plot(track_d_loss, alpha=0.3, linewidth=5)
    axs[0].plot(xs[w:-w], smooth(track_d_loss, w)[w:-w], c='C0')
    axs[0].set_title('Discriminator', fontsize=10)
    axs[0].set_yscale('log')
    axs[0].set_ylabel('loss', fontsize=10)
    plt.xkcd()
    axs[1].plot(track_g_loss, alpha=0.3, linewidth=5, c='C4')
    axs[1].plot(xs[w:-w], smooth(track_g_loss, w)[w:-w], c='C4')
    axs[1].set_title('Generator', fontsize=10)
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('loss', fontsize=10)
    plt.savefig(output_dir + f'/loss_trackinge.png')
    plt.close()


def get_device(use_cuda):
    import torch

    device = "cpu"
    num_gpus = 0
        
    if torch.cuda.is_available():
        if use_cuda:
            device = "cuda"
            torch.cuda.set_device(0)
            num_gpus = torch.cuda.device_count()
        else:
            logger.debug("WARNING: You have a CUDA device, so you should probably run with --cuda 1")

    logger.debug(f"Number of gpus available: {num_gpus}")
    
    return device, num_gpus


def train(dataloader, hps, batch_size, test_batch_size, epochs, learning_rate,
          device, hosts, backend, current_host, model_dir, output_dir, seed,
          log_interval, sample_interval,
          beta1, nz, nc, ngf, ndf):
        
    dcgan = DCGAN(batch_size=batch_size, nz=nz, nc=nc, ngf=ngf, ndf=ndf,
                    device=device, weights_init=weights_init, learning_rate=learning_rate,
                    betas=(beta1, 0.999), real_label=1, fake_label=0)

    track_d_loss = []
    track_g_loss = []

    for epoch in range(epochs):
        batches = len(dataloader)
        for batch, data in enumerate(dataloader, 0):
            errG, errD, D_x, D_G_z1, D_G_z2 = dcgan.train_step(data,
                          epoch=epoch, epochs=epochs, batch=batch, batches=batches)

            track_g_loss.append(errG)
            track_d_loss.append(errD)
                
            log_batch(epoch, epochs, batch, batches, errD, errG,
                            D_x, D_G_z1, D_G_z2, log_interval=log_interval, output_dir=output_dir)
            sample_batch(dcgan, epoch, batch, output_dir=output_dir,
                                sample_interval=sample_interval)

        # do checkpointing
        checkpoint_epoch(dcgan, epoch, output_dir)
        
    save_model(model_dir, dcgan.netG)
    save_track_loss(track_d_loss, track_d_loss, output_dir=output_dir)

    return


def save_model(model_dir, model):
    logger.info("Saving the model.")
    model.save(model_dir, filename="generator_state.pth")

    
def load_model(model_dir, device=None):
    logger.info("Loading the model.")
    if device is None:
        device = get_training_device_name(1)

    netG.load(model_dir, filename="generator_state.pth", device=device)

    return netG


def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Example')
    
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=None, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--learning-rate', type=float, default=0.0002, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--seed', type=int, default=None, metavar='S',
                        help='random seed')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--sample-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before sampling training model output')
    parser.add_argument('--backend', type=str, default=None,
                        help='backend for distributed training (tcp, gloo on cpu and gloo, nccl on gpu)')

    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')

    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', None))

    parser.add_argument('--current-host', type=str, default=os.environ.get('SM_CURRENT_HOST', None))
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ.get('SM_HOSTS', "{}")))
    
    parser.add_argument('--cuda', type=int, default=1)
    parser.add_argument('--num-gpus', type=int, default=os.environ.get('SM_NUM_GPUS', None))
    
    parser.add_argument('--dataset', required=True, help='cifar10 | lsun | mnist | qmnist |imagenet | folder | lfw | fake')
    parser.add_argument('--pin-memory', type=bool, default=os.environ.get('SM_PIN_MEMORY', False))

    parser.add_argument('--data-dir', required=False, default=None, help='path to data dir')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
    parser.add_argument('--image-size', type=int, default=64, help='the height / width of the input image to network')
    parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--output-dir', default=os.environ.get('SM_OUTPUT_DATA_DIR', None), help='folder to output images and model checkpoints')
    parser.add_argument('--hps', default=os.environ.get('SM_HPS', None), help='Hyperparameters')
    
    return parser.parse_known_args()


def get_datasets(dataset_name, *, dataroot='/opt/ml/input/data', image_size, classes=None):

    logger.info(f"dataname: {dataset_name}, dataroot: {dataroot}, " +
                    f"image_size: {image_size}")
    if dataset_name in ['imagenet', 'folder', 'lfw']:
        # folder dataset
        dataset = dset.ImageFolder(root=dataroot,
                                   transform=transforms.Compose([
                                       transforms.Resize(image_size),
                                       transforms.CenterCrop(image_size),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   ]))
    elif dataset_name == 'lsun':
        classes = [ c + '_train' for c in args.classes.split(',')]
        dataset = dset.LSUN(root=dataroot, classes=classes,
                            transform=transforms.Compose([
                                transforms.Resize(image_size),
                                transforms.CenterCrop(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))
        nc=3
    elif dataset_name == 'cifar10':
        dataset = dset.CIFAR10(root=dataroot,
                               transform=transforms.Compose([
                                   transforms.Resize(image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
        nc=3

    elif dataset_name == 'mnist':
        dataset = dset.MNIST(root=dataroot,
                               transform=transforms.Compose([
                                   transforms.Resize(image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5,), (0.5,)),
                               ]))
        nc=1

    elif dataset_name == 'qmnist':
        dataset = dset.QMNIST(root=dataroot,
                               transform=transforms.Compose([
                                   transforms.Resize(image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5,), (0.5,)),
                               ]))
        nc=1
    
    elif dataset_name == 'fake':
        dataset = dset.FakeData(image_size=(3, image_size, image_size),
                                transform=transforms.ToTensor())
        nc=3

    return dataset, nc
    

if __name__ == '__main__':
    args, unknown = parse_args()
    
    # get training options
    hps = json.loads(args.hps)

    try:
        os.makedirs(args.output_dir)
    except OSError:
        pass

    if args.seed is None:
        random_seed = random.randint(1, 10000)
        logger.debug(f"Generated Random Seed: {random_seed}")
        cudnn.benchmark = True
    else:
        logger.debug(f"Provided Random Seed: {args.seed}")
        random_seed = args.seed
        cudnn.deterministic = True
        cudnn.benchmark = False
        
    random.seed(random_seed)
    torch.manual_seed(random_seed)

    pin_memory=args.pin_memory
    num_workers = int(args.workers)
    
    device, num_gpus = get_device(args.cuda)
    
    if device == 'cuda':
        num_workers = 1
        pin_memory = True


#     if args.distributed:
#         # Initialize the distributed environment.
#         world_size = len(args.hosts)
#         os.environ['WORLD_SIZE'] = str(world_size)
#         host_rank = args.hosts.index(args.current_host)
#         dist.init_process_group(backend=args.backend, rank=host_rank)
    
    if args.data_dir is None:
        input_dir = os.environ.get('SM_INPUT_DIR', None)
        if input_dir is None and str(args.dataset).lower() != 'fake':
            raise ValueError(f"`--data-dir` parameter is required for dataset \"{args.dataset}\"")

        dataroot = input_dir + "/data"
    else:
        dataroot = args.data_dir

    dataset, nc = get_datasets(args.dataset, dataroot=dataroot, image_size=args.image_size)

    assert dataset
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                    shuffle=True, num_workers=num_workers, pin_memory=pin_memory)

#     nc = int(args.nc)
    nz = int(args.nz)
    ngf = int(args.ngf)
    ndf = int(args.ndf)

    
    if args.epochs is None:
        epochs = hps.get('epochs', 10)
    else:
        epochs = args.epochs
    
    train(dataloader, hps, args.batch_size, args.test_batch_size, epochs, args.learning_rate,
          device, args.hosts, args.backend, args.current_host,
          args.model_dir, args.output_dir, args.seed, args.log_interval, args.sample_interval, args.beta1,
          nz, nc, ngf, ndf)

