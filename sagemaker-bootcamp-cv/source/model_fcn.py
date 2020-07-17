# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
import argparse
import json
import logging
import sys
import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision
import torchvision.models
import torchvision.transforms as transforms

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('/opt/ml/output/tensorboard/')

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# logger.addHandler(logging.StreamHandler(sys.stdout))

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# https://pytorch.org/tutorials/beginner/pytorch_with_examples.html
class FCNNet(nn.Module):
    def __init__(self, D_in, H1, H2, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(FCNNet, self).__init__()
        self.fc1 = torch.nn.Linear(D_in, H1)
        self.fc2 = torch.nn.Linear(H1, H2)
        self.fc3 = torch.nn.Linear(H2, D_out)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        x = x.view(-1, 3 * 32 * 32)
        h1_relu = self.fc1(x).clamp(min=0)
        h2_relu = self.fc2(h1_relu).clamp(min=0)
        y_pred = self.fc3(h2_relu)
        return y_pred


def _train(args):
    is_distributed = len(args.hosts) > 1 and args.dist_backend is not None
    logger.debug('Distributed training - {}'.format(is_distributed))

    if is_distributed:
        # Initialize the distributed environment.
        world_size = len(args.hosts)
        os.environ['WORLD_SIZE'] = str(world_size)
        host_rank = args.hosts.index(args.current_host)
        os.environ['RANK'] = str(host_rank)
        dist.init_process_group(backend=args.dist_backend, rank=host_rank, world_size=world_size)
        logger.info(
            'Initialized the distributed environment: \'{}\' backend on {} nodes. '.format(
                args.dist_backend,
                dist.get_world_size()) + 'Current host rank is {}. Using cuda: {}. Number of gpus: {}'.format(
                dist.get_rank(), torch.cuda.is_available(), args.num_gpus))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info('Device Type: {}'.format(device))

    logger.info('Loading Cifar10 dataset')
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root=args.data_dir, train=True,
                                            download=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.bs,
                                               shuffle=True, num_workers=args.workers)

    logger.info('Model loaded')
    model = FCNNet(3072, 1024, 256, 10)

    if torch.cuda.device_count() > 1:
        logger.info('Gpu count: {}'.format(torch.cuda.device_count()))
        model = nn.DataParallel(model)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(0, args.epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            # get the inputs
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            statistics = f'[{epoch + 1:3d}, {i + 1:5d}] train_loss: {running_loss / 200:.3f};'
            logger.info(statistics)
            if i % 200 == 199:  # print every 200 mini-batches
                print(statistics)
                running_loss = 0.0

#             writer.add_scalar('training/loss', loss.item(), epoch + 1)
            
#             import torch.cuda as cutorch

#             for i in range(cutorch.device_count()):
#                 writer.add_scalar(f'GPU/{i}', cutorch.memory_allocated(i), epoch + 1)


    print('Finished Training')
    return _save_model(model, args.model_dir)


def _save_model(model, model_dir):
    logger.info('Saving the model.')
    path = os.path.join(model_dir, 'model.pth')
    # recommended way from http://pytorch.org/docs/master/notes/serialization.html
    torch.save(model.cpu().state_dict(), path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--workers', type=int, default=2, metavar='W',
                        help='number of data loading workers (default: 2)')
    parser.add_argument('--epochs', type=int, default=2, metavar='E',
                        help='number of total epochs to run (default: 2)')
    parser.add_argument('--bs', type=int, default=4, metavar='BS',
                        help='batch size (default: 4)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='initial learning rate (default: 0.001)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--dist_backend', type=str, default='gloo', help='distributed backend (default: gloo)')

    parser.add_argument('--hosts', type=json.loads, default=os.environ['SM_HOSTS'])
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--num-gpus', type=int, default=os.environ['SM_NUM_GPUS'])

    _train(parser.parse_args())


def model_fn(model_dir):
    logger.info('model_fn')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = FCNNet(3072, 1024, 256, 10)
    if torch.cuda.device_count() > 1:
        logger.info('Gpu count: {}'.format(torch.cuda.device_count()))
        model = nn.DataParallel(model)

    with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
        model.load_state_dict(torch.load(f))
    return model.to(device)
