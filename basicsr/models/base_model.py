"""Minimal distributed training services required by experiment 6220553."""

import os
from collections import OrderedDict

import torch
from torch.nn.parallel import DistributedDataParallel

from basicsr.models.lr_scheduler import CosineAnnealingRestartLR
from basicsr.utils import get_root_logger
from basicsr.utils.dist_util import master_only


class BaseModel:
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device('cuda')
        self.schedulers = []
        self.optimizers = []

    def validation(self, dataloader, current_iter, tb_logger):
        self.dist_validation(dataloader, current_iter, tb_logger)

    def model_ema(self, decay):
        source = dict(self.get_bare_model(self.net_g).named_parameters())
        target = dict(self.net_g_ema.named_parameters())
        for name, parameter in target.items():
            parameter.data.mul_(decay).add_(
                source[name].data, alpha=1.0 - decay)

    def get_current_log(self):
        return self.log_dict

    def model_to_device(self, network):
        network = network.to(self.device)
        return DistributedDataParallel(
            network, device_ids=[torch.cuda.current_device()])

    def setup_schedulers(self):
        options = dict(self.opt['train']['scheduler'])
        self.schedulers = [
            CosineAnnealingRestartLR(optimizer, **options)
            for optimizer in self.optimizers
        ]

    @staticmethod
    def get_bare_model(network):
        return network.module if isinstance(network, DistributedDataParallel) else network

    @master_only
    def print_network(self, network):
        bare = self.get_bare_model(network)
        parameters = sum(parameter.numel() for parameter in bare.parameters())
        get_root_logger().info(
            f'Network: {bare.__class__.__name__}, with parameters: {parameters:,d}')

    def update_learning_rate(self, current_iter):
        if current_iter > 1:
            for scheduler in self.schedulers:
                scheduler.step()

    def get_current_learning_rate(self):
        return [group['lr'] for group in self.optimizer_g.param_groups]

    @master_only
    def save_network(self, networks, label, current_iter, param_key):
        iteration = 'latest' if current_iter == -1 else current_iter
        path = os.path.join(self.opt['path']['models'], f'{label}_{iteration}.pth')
        state = {}
        for network, key in zip(networks, param_key):
            state[key] = {
                name: value.detach().cpu()
                for name, value in self.get_bare_model(network).state_dict().items()
            }
        torch.save(state, path)

    @master_only
    def save_training_state(self, epoch, current_iter):
        if current_iter == -1:
            return
        state = {
            'epoch': epoch,
            'iter': current_iter,
            'optimizers': [optimizer.state_dict() for optimizer in self.optimizers],
            'schedulers': [scheduler.state_dict() for scheduler in self.schedulers],
        }
        path = os.path.join(
            self.opt['path']['training_states'], f'{current_iter}.state')
        torch.save(state, path)

    def reduce_loss_dict(self, loss_dict):
        with torch.no_grad():
            keys = tuple(loss_dict)
            values = torch.stack(tuple(loss_dict.values()))
            torch.distributed.reduce(values, dst=0)
            if self.opt['rank'] == 0:
                values /= self.opt['world_size']
            return OrderedDict(
                (name, value.mean().item())
                for name, value in zip(keys, values))
