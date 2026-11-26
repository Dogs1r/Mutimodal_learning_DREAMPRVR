# -*- coding: utf-8 -*-
import os
import argparse
from pathlib import Path
import yaml
import numpy as np
import random
import sys
import time
from tqdm import tqdm
import ipdb
import pickle

import torch
import torch.nn as nn

from Configs.builder import get_configs
from Models.builder import get_models
from Datasets.builder import get_datasets
from Opts.builder import get_opts
from Losses.builder import get_losses
from Validations.builder import get_validations

from Utils.basic_utils import AverageMeter, BigFile, read_dict, log_config
from Utils.utils import set_seed, set_log, gpu, save_ckpt, load_ckpt


root_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_path)

parser = argparse.ArgumentParser(description="Partially Relevant Video Retrieval")
parser.add_argument(
    '-d', '--dataset_name', default='cha', type=str, metavar='DATASET', help='dataset name', 
    choices=['tvr', 'act', 'cha']
)
parser.add_argument(
    '--gpu', default = '0', type = str, help = 'specify gpu device'
    )
parser.add_argument('--eval', action='store_true')
parser.add_argument('--resume', default='', type=str)
parser.add_argument('--experiment', choices=['baseline', 'fixed-mean', 'fixed-sentence',
                                             'learned-mean', 'learned-sentence'])
parser.add_argument('--run-name')
parser.add_argument('--epochs', type=int)
parser.add_argument('--batch-size', type=int)
parser.add_argument('--seed', type=int)
parser.add_argument('--eval-dev', action='store_true')


def train_one_epoch(epoch, train_loader, model, criterion, cfg, optimizer):

    if epoch >= cfg['hard_negative_start_epoch']:
        criterion.cfg['use_hard_negative'] = True
    else:
        criterion.cfg['use_hard_negative'] = False

    loss_meter = AverageMeter()
    loss_diffusion_meter = AverageMeter()
    model.train()

    train_bar = tqdm(train_loader, desc="epoch " + str(epoch), total=len(train_loader),
                    unit="batch", dynamic_ncols=True)

    for idx, batch in enumerate(train_bar):

        batch = gpu(batch)

        optimizer.zero_grad()

        input_list = model(batch)

        loss,loss_diffusion = criterion(input_list, batch)
        
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at epoch {epoch}, batch {idx}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"), error_if_nonfinite=True)
        optimizer.step()

        loss_meter.update(loss.cpu().item())
        loss_diffusion_meter.update(loss_diffusion.cpu().item())
        train_bar.set_description('exp: {} epoch:{:2d} iter:{:3d} loss:{:.4f} loss_diffusion:{:.4f}'.format(cfg['model_name'], epoch, idx, loss,loss_diffusion))
    return loss_meter.avg,loss_diffusion_meter.avg


def val_one_epoch(epoch, context_dataloader, query_eval_loader, model, val_criterion, cfg, optimizer, best_val, loss_meter ,logger,loss_diffusion_meter=None):

    val_meter = val_criterion(model, context_dataloader, query_eval_loader,
                              text_t2t=not cfg.get('development_split', False))

    improved = (val_meter[0], val_meter[4]) > (best_val[0], best_val[4]) if cfg.get('development_split') else val_meter[4] > best_val[4]
    if improved:
        es = False
        sc = 'New Best Model !!!'
        best_val = val_meter
        save_ckpt(model, optimizer, cfg, os.path.join(cfg['model_root'], 'best.ckpt'), epoch, best_val)
    else:
        es = True
        sc = 'A Relative Failure Epoch'
                
    logger.info('==========================================================================================================')
    logger.info('Epoch: {:2d}    {}'.format(epoch, sc))
    if loss_diffusion_meter:
        logger.info('Average diffusion Loss: {:.4f}'.format(loss_diffusion_meter))       
    logger.info('Average Loss: {:.4f}'.format(loss_meter))
    logger.info('R@1: {:.1f}'.format(val_meter[0]))
    logger.info('R@5: {:.1f}'.format(val_meter[1]))
    logger.info('R@10: {:.1f}'.format(val_meter[2]))
    logger.info('R@100: {:.1f}'.format(val_meter[3]))
    logger.info('Rsum: {:.1f}'.format(val_meter[4]))
    logger.info('Best: R@1: {:.1f} R@5: {:.1f} R@10: {:.1f} R@100: {:.1f} Rsum: {:.1f}'.format(best_val[0], best_val[1], best_val[2], best_val[3], best_val[4]))
    logger.info('==========================================================================================================')

    return val_meter, best_val, es



def validation(context_dataloader, query_eval_loader, model, val_criterion, cfg, logger, resume):

    val_meter = val_criterion(model, context_dataloader, query_eval_loader,
                              text_t2t=not cfg.get('development_split', False))
    
    logger.info('==========================================================================================================')
    logger.info('Testing from: {}'.format(resume))
    logger.info('R@1: {:.1f}'.format(val_meter[0]))
    logger.info('R@5: {:.1f}'.format(val_meter[1]))
    logger.info('R@10: {:.1f}'.format(val_meter[2]))
    logger.info('R@100: {:.1f}'.format(val_meter[3]))
    logger.info('Rsum: {:.1f}'.format(val_meter[4]))
    logger.info('==========================================================================================================')


def main():
    args = parser.parse_args()
    if args.eval_dev and not args.experiment:
        parser.error("--eval-dev needs --experiment")
    if (args.eval or args.eval_dev) and not args.resume:
        parser.error("evaluation needs --resume")
    if args.experiment:
        if args.dataset_name != 'cha' or not args.run_name or Path(args.run_name).name != args.run_name or args.run_name in ('.', '..'):
            parser.error("Charades experiments need --run-name NAME (one directory name)")
        if args.epochs is not None and args.epochs < 1 or args.batch_size is not None and args.batch_size < 2:
            parser.error("epochs must be positive and batch size at least two")
        from Configs.cha import cfg as defaults
        root = Path(defaults['root']) / 'AdaptivePRVR' / args.run_name
        root.mkdir(parents=True, exist_ok=bool(args.resume))
        cfg = dict(defaults, model_root=str(root), ckpt_path=str(root), model_name=args.experiment,
                   event_variant=args.experiment, development_split=True,
                   event_temperature=0.1, event_ce_temperature=0.07, event_loss_weight=0.1)
        for key, value in [('seed', args.seed), ('batchsize', args.batch_size), ('n_epoch', args.epochs)]:
            if value is not None:
                cfg[key] = value
        if not args.resume:
            (root / 'hyperparams.yaml').write_text(yaml.safe_dump(cfg), encoding='utf-8')
    else:
        if any(v is not None for v in (args.run_name, args.epochs, args.batch_size, args.seed)):
            parser.error("experimental run options need --experiment")
        cfg = get_configs(args.dataset_name)


    # set logging
    logger = set_log(cfg['model_root'], 'log.txt')
    logger.info('Partially Relevant Video Retrieval Training: {}'.format(cfg['dataset_name']))

    # set seed
    set_seed(cfg['seed'])
    logger.info('set seed: {}'.format(cfg['seed']))

    # hyper parameter
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device_ids = range(torch.cuda.device_count())
    logger.info('used gpu: {}'.format(args.gpu))

    logger.info('Hyper Parameter ......')
    logger.info(cfg)

    # dataset
    logger.info('Loading Data ......')
    cfg, train_loader, context_dataloader, query_eval_loader, test_context_dataloader, test_query_eval_loader = get_datasets(cfg)

    # model
    logger.info('Loading Model ......') 
    model = get_models(cfg)
    
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

    print(f"Total Parameters: {total_params:.2f}M")
    print(f"Trainable Parameters: {trainable_params:.2f}M")
    
    # initial
    current_epoch = -1
    es_cnt = 0
    best_val = [0., 0., 0., 0., 0.]
    if args.resume != '':
        logger.info('Resume from {}'.format(args.resume))
        saved, model_state_dict, optimizer_state_dict, current_epoch, best_val = load_ckpt(args.resume)
        if args.experiment and saved.get('event_variant') != args.experiment:
            parser.error("checkpoint experiment does not match --experiment")
        if args.experiment and not (args.eval or args.eval_dev):
            for key in ('seed', 'n_epoch', 'batchsize'):
                if cfg[key] != saved[key]:
                    parser.error(f"resume needs the original {key}={saved[key]}")
        model.load_state_dict(model_state_dict)
    model = model.cuda()
    if len(device_ids) > 1:
        model = nn.DataParallel(model)
    
    criterion = get_losses(cfg)
    val_criterion = get_validations(cfg)

    if args.eval or args.eval_dev:
        if args.resume == '':
            logger.info('No trained ckpt load !!!') 
        else:
            with torch.no_grad():
                loaders = (context_dataloader, query_eval_loader) if args.eval_dev else (test_context_dataloader, test_query_eval_loader)
                validation(*loaders, model, val_criterion, cfg, logger, args.resume)
        exit(0)

    optimizer = get_opts(cfg, model, train_loader)
    if args.resume != '':
        optimizer.load_state_dict(optimizer_state_dict)

    for epoch in range(current_epoch + 1, cfg['n_epoch']):

        ############## train
        loss_meter,loss_diffusion_meter = train_one_epoch(epoch, train_loader, model, criterion, cfg, optimizer)

        ############## val
        with torch.no_grad():
            val_meter, best_val, es = val_one_epoch(epoch, context_dataloader, query_eval_loader, model, 
                    val_criterion, cfg, optimizer, best_val, loss_meter, logger,loss_diffusion_meter)

        ############## early stop
        if not es:
            es_cnt = 0
        else:
            es_cnt += 1
            if cfg['max_es_cnt'] != -1 and es_cnt > cfg['max_es_cnt']: 
                logger.info('Early Stop !!!') 
                exit(0)


if __name__ == '__main__':
    main()
