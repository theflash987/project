"""Logging used by NoRouter-K16."""

import datetime
import logging
import time

from .dist_util import get_dist_info, master_only

_INITIALIZED = set()


class AvgTimer:
    def __init__(self, window=200):
        self.window = window
        self.start()
        self.count = 0
        self.total = 0.0
        self.average = 0.0

    def start(self):
        self.tick = time.time()

    def record(self):
        elapsed = time.time() - self.tick
        self.count += 1
        self.total += elapsed
        self.average = self.total / self.count
        if self.count > self.window:
            self.count = 0
            self.total = 0.0
        self.tick = time.time()

    def get_avg_time(self):
        return self.average


class MessageLogger:
    def __init__(self, opt, start_iter, tb_logger):
        self.name = opt['name']
        self.total_iterations = opt['train']['total_iter']
        self.tb_logger = tb_logger
        self.start_iter = start_iter
        self.start_time = time.time()
        self.logger = get_root_logger()

    def reset_start_time(self):
        self.start_time = time.time()

    @master_only
    def __call__(self, values):
        epoch = values.pop('epoch')
        iteration = values.pop('iter')
        rates = values.pop('lrs')
        iteration_time = values.pop('time')
        data_time = values.pop('data_time')
        average = (time.time() - self.start_time) / (iteration - self.start_iter + 1)
        eta = datetime.timedelta(
            seconds=int(average * (self.total_iterations - iteration - 1)))
        message = (
            f'[{self.name[:5]}..][epoch:{epoch:3d}, iter:{iteration:8,d}, '
            f'lr:({"".join(f"{rate:.3e}," for rate in rates)})] '
            f'[eta: {eta}, time (data): {iteration_time:.3f} ({data_time:.3f})] ')
        for key, value in values.items():
            message += f'{key}: {value:.4e} '
            if self.tb_logger:
                group = 'losses/' if key.startswith('l_') else ''
                self.tb_logger.add_scalar(group + key, value, iteration)
        self.logger.info(message)


@master_only
def init_tb_logger(log_dir):
    from torch.utils.tensorboard import SummaryWriter
    return SummaryWriter(log_dir=log_dir)


def get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=None):
    logger = logging.getLogger(logger_name)
    if logger_name in _INITIALIZED:
        return logger
    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    logger.propagate = False
    rank, _ = get_dist_info()
    logger.setLevel(logging.ERROR if rank else log_level)
    if rank == 0 and log_file:
        file_handler = logging.FileHandler(log_file, 'w')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    _INITIALIZED.add(logger_name)
    return logger


def get_env_info():
    import torch
    return f'PyTorch: {torch.__version__}; CUDA: {torch.version.cuda}'
