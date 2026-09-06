"""TimePoleFusion continuation from saved weights, EMA, Adam and cosine state."""

import datetime
import logging
import math
import os
import time

import torch
from torch import distributed as dist

from basicsr.data import build_dataloader, build_dataset
from basicsr.data.data_sampler import EnlargedSampler
from basicsr.data.prefetch_dataloader import CPUPrefetcher
from basicsr.models import build_model
from basicsr.utils import (
    AvgTimer,
    MessageLogger,
    get_env_info,
    get_root_logger,
    get_time_str,
    init_tb_logger,
    make_exp_dirs,
)
from basicsr.utils.options import copy_opt_file, dict2str, parse_options


def restore_training(model, resume):
    state = torch.load(resume['state'], map_location='cpu', weights_only=False)
    weights = torch.load(resume['model'], map_location='cpu', weights_only=True)
    assert state['iter'] == resume['expected_iter']
    model.get_bare_model(model.net_g).load_state_dict(weights['params'], strict=True)
    model.net_g_ema.load_state_dict(weights['params_ema'], strict=True)
    assert len(model.optimizers) == len(state['optimizers'])
    assert len(model.schedulers) == len(state['schedulers'])
    for optimizer, saved in zip(model.optimizers, state['optimizers']):
        optimizer.load_state_dict(saved)
    for scheduler, saved in zip(model.schedulers, state['schedulers']):
        scheduler.load_state_dict(saved)
        assert scheduler.last_epoch == state['iter'] - 1
    get_root_logger().info(
        'TRUE_RESUME iter=%s epoch=%s scheduler_epoch=%s lr=%s optimizer_states=%s; '
        'strict generator/EMA/buffers restored; source checkpoint has no RNG state.',
        state['iter'], state['epoch'], model.schedulers[0].last_epoch,
        model.get_current_learning_rate(), len(model.optimizer_g.state))
    return state


def train_pipeline(root_path):
    job_start_time = time.monotonic()
    opt, args = parse_options(root_path)
    torch.backends.cudnn.benchmark = True
    make_exp_dirs(opt)
    copy_opt_file(args.opt, opt['path']['experiments_root'])

    log_path = os.path.join(
        opt['path']['log'], f"train_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(log_level=logging.INFO, log_file=log_path)
    logger.info(get_env_info())
    logger.info(dict2str(opt))
    tb_logger = init_tb_logger(
        os.path.join(root_path, 'tb_logger', opt['name']))

    train_opt = opt['datasets']['train']
    train_set = build_dataset(train_opt)
    sampler = EnlargedSampler(
        train_set,
        opt['world_size'],
        opt['rank'],
        train_opt['dataset_enlarge_ratio'])
    train_loader = build_dataloader(
        train_set, train_opt, sampler=sampler, seed=opt['manual_seed'])
    val_opt = opt['datasets']['val']
    val_loader = build_dataloader(build_dataset(val_opt), val_opt)

    iterations_per_epoch = len(train_loader)
    total_iterations = int(opt['train']['total_iter'])
    total_epochs = math.ceil(total_iterations / iterations_per_epoch)
    logger.info(
        f'Training samples: {len(train_set)}; iterations/epoch: '
        f'{iterations_per_epoch}; epochs: {total_epochs}; '
        f'iterations: {total_iterations}.')

    model = build_model(opt)
    resume_state = restore_training(model, opt['resume'])
    start_iter = int(resume_state['iter'])
    start_epoch, skipped_batches = divmod(start_iter, iterations_per_epoch)
    assert (start_iter - 1) // iterations_per_epoch == resume_state['epoch']
    logger.info('Resume data position: epoch=%s completed_batches=%s steps_per_epoch=%s',
                start_epoch, skipped_batches, iterations_per_epoch)
    message_logger = MessageLogger(opt, start_iter, tb_logger)
    prefetcher = CPUPrefetcher(train_loader)
    data_timer = AvgTimer()
    iter_timer = AvgTimer()
    start_time = time.time()
    current_iter = start_iter
    stop_requested = torch.zeros((), device=model.device, dtype=torch.int32)

    for epoch in range(start_epoch, total_epochs + 1):
        sampler.set_epoch(epoch)
        prefetcher.reset()
        if epoch == start_epoch:
            for _ in range(skipped_batches):
                prefetcher.next()
        train_data = prefetcher.next()
        while train_data is not None and current_iter < total_iterations:
            data_timer.record()
            current_iter += 1
            model.update_learning_rate(current_iter)
            model.feed_data(train_data)
            model.optimize_parameters(current_iter)
            iter_timer.record()
            if current_iter == start_iter + 1:
                message_logger.reset_start_time()
            if current_iter == start_iter + 1 or current_iter % opt['logger']['print_freq'] == 0:
                values = {
                    'epoch': epoch,
                    'iter': current_iter,
                    'lrs': model.get_current_learning_rate(),
                    'time': iter_timer.get_avg_time(),
                    'data_time': data_timer.get_avg_time(),
                    **model.get_current_log(),
                }
                message_logger(values)
            if current_iter % opt['logger']['save_checkpoint_freq'] == 0:
                logger.info('Saving models and training state.')
                model.save(epoch, current_iter)
            if current_iter % opt['val']['val_freq'] == 0:
                model.validation(val_loader, current_iter, tb_logger)
            if current_iter % 100 == 0:
                stop_requested.fill_(time.monotonic() - job_start_time >= opt['resume']['max_seconds'])
                dist.all_reduce(stop_requested, op=dist.ReduceOp.MAX)
                if stop_requested.item():
                    logger.info('WALLTIME_SAVE_STOP iter=%s', current_iter)
                    break
            data_timer.start()
            iter_timer.start()
            train_data = prefetcher.next()
        if current_iter == total_iterations or stop_requested.item():
            break

    elapsed = datetime.timedelta(seconds=int(time.time() - start_time))
    logger.info(f'End of training. Time consumed: {elapsed}')
    model.save(epoch=epoch, current_iter=current_iter)
    model.save(epoch=-1, current_iter=-1)
    model.validation(val_loader, current_iter, tb_logger)
    if tb_logger:
        tb_logger.close()
    logger.info('TIME_POLE_RESUME_COMPLETE iter=%s', current_iter)
    dist.destroy_process_group()


if __name__ == '__main__':
    train_pipeline(os.path.abspath(os.path.join(__file__, '..', '..')))
