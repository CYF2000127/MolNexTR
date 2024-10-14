import os
import sys
import time
import json
import random
import argparse
import datetime
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.optim import AdamW
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import get_scheduler
from MolNexTR.dataset import TrainDataset, AuxTrainDataset, bms_collate
from MolNexTR.components import Encoder, Decoder
from MolNexTR.loss_fuc import Criterion
from MolNexTR.utils import seed_torch, save_args, init_summary_writer, LossMeter, AverageMeter, asMinutes, timeSince, \
    print_rank_0, format_df
from MolNexTR.chemical import convert_graph_to_smiles, postprocess_smiles, keep_main_molecule
from MolNexTR.tokenization import get_tokenizer
from evaluate import SmilesEvaluator

import warnings
warnings.filterwarnings('ignore')


def get_args():
    """
    Parse command-line arguments to configure the model's training, evaluation, and testing procedures.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--do_train', action='store_true', help='Flag for training mode')
    parser.add_argument('--do_valid', action='store_true', help='Flag for validation mode')
    parser.add_argument('--do_test', action='store_true', help='Flag for testing mode')
    parser.add_argument('--fp16', action='store_true', help='Use half-precision floating-point format for training')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--print_freq', type=int, default=200, help='Frequency for printing training progress')
    parser.add_argument('--debug', action='store_true', help='Enable debugging mode')
    parser.add_argument('--backend', type=str, default='gloo', choices=['gloo', 'nccl'], help='Backend for distributed training')
    
    # Model parameters
    parser.add_argument('--encoder', type=str, default='swin_base', help='Type of encoder to use')
    parser.add_argument('--decoder', type=str, default='lstm', help='Type of decoder to use')
    parser.add_argument('--no_pretrained', action='store_true', help='Disable pretrained weights')
    parser.add_argument('--use_checkpoint', action='store_true', help='Use checkpoint for training')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate for the model')
    parser.add_argument('--embed_dim', type=int, default=256, help='Embedding dimension for the model')
    parser.add_argument('--enc_pos_emb', action='store_true', help='Enable positional embeddings in encoder')

    # Model-specific options
    group = parser.add_argument_group("lstm_options")
    group.add_argument('--decoder_dim', type=int, default=512, help='Hidden size for LSTM decoder')
    group.add_argument('--decoder_layer', type=int, default=1, help='Number of layers for LSTM decoder')
    group.add_argument('--attention_dim', type=int, default=256, help='Attention dimension for LSTM decoder')

    group = parser.add_argument_group("transformer_options")
    group.add_argument("--dec_num_layers", type=int, default=6, help='Number of layers in transformer decoder')
    group.add_argument("--dec_hidden_size", type=int, default=256, help='Hidden size for transformer decoder')
    group.add_argument("--dec_attn_heads", type=int, default=8, help='Number of attention heads in transformer decoder')
    group.add_argument("--dec_num_queries", type=int, default=128, help='Number of query vectors in transformer decoder')
    group.add_argument("--hidden_dropout", type=float, default=0.1, help='Hidden layer dropout rate')
    group.add_argument("--attn_dropout", type=float, default=0.1, help='Attention dropout rate')
    group.add_argument("--max_relative_positions", type=int, default=0, help='Max relative positions for attention mechanism')

    # Data parameters
    parser.add_argument('--data_path', type=str, default=None, help='Path to data directory')
    parser.add_argument('--train_file', type=str, default=None, help='File name for training data')
    parser.add_argument('--valid_file', type=str, default=None, help='File name for validation data')
    parser.add_argument('--test_file', type=str, default=None, help='File name for testing data')
    parser.add_argument('--aux_file', type=str, default=None, help='File name for auxiliary data')
    parser.add_argument('--coords_file', type=str, default=None, help='File containing coordinates')
    parser.add_argument('--vocab_file', type=str, default=None, help='File containing vocabulary data')
    parser.add_argument('--dynamic_indigo', action='store_true', help='Enable dynamic Indigo transformations')
    parser.add_argument('--default_option', action='store_true', help='Use default Indigo options')
    parser.add_argument('--pseudo_coords', action='store_true', help='Use pseudo coordinates for atoms')
    parser.add_argument('--include_condensed', action='store_true', help='Include condensed molecular representations')
    parser.add_argument('--formats', type=str, default=None, help='Data formats for training')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for data loading')
    parser.add_argument('--input_size', type=int, default=384, help='Input size for image data')
    parser.add_argument('--multiscale', action='store_true', help='Enable multi-scale processing')
    parser.add_argument('--augment', action='store_true', help='Enable data augmentation')
    parser.add_argument('--mol_augment', action='store_true', help='Enable molecular augmentation')
    parser.add_argument('--coord_bins', type=int, default=100, help='Number of bins for coordinates')
    parser.add_argument('--sep_xy', action='store_true', help='Separate x and y coordinates')
    parser.add_argument('--mask_ratio', type=float, default=0, help='Ratio for masking coordinates during training')
    parser.add_argument('--continuous_coords', action='store_true', help='Use continuous coordinates')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=8, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training')
    parser.add_argument('--encoder_lr', type=float, default=1e-4, help='Learning rate for encoder')
    parser.add_argument('--decoder_lr', type=float, default=4e-4, help='Learning rate for decoder')
    parser.add_argument('--weight_decay', type=float, default=1e-6, help='Weight decay for optimizer')
    parser.add_argument('--max_grad_norm', type=float, default=5., help='Max gradient norm for clipping')
    parser.add_argument('--scheduler', type=str, choices=['cosine', 'constant'], default='cosine', help='Scheduler type for learning rate')
    parser.add_argument('--warmup_ratio', type=float, default=0, help='Warmup ratio for learning rate scheduler')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Number of gradient accumulation steps')
    parser.add_argument('--load_path', type=str, default=None, help='Path to load pre-trained model weights')
    parser.add_argument('--load_encoder_only', action='store_true', help='Load encoder weights only')
    parser.add_argument('--train_steps_per_epoch', type=int, default=-1, help='Number of training steps per epoch')
    parser.add_argument('--save_path', type=str, default='output/', help='Path to save model checkpoints')
    parser.add_argument('--save_mode', type=str, default='best', choices=['best', 'all', 'last'], help='Mode for saving checkpoints')
    parser.add_argument('--load_ckpt', type=str, default='best', help='Checkpoint to load')
    parser.add_argument('--resume', action='store_true', help='Resume training from last checkpoint')
    parser.add_argument('--all_data', action='store_true', help='Use all data for training')
    parser.add_argument('--init_scheduler', action='store_true', help='Initialize scheduler on training resumption')
    parser.add_argument('--label_smoothing', type=float, default=0.0, help='Label smoothing for training')
    parser.add_argument('--shuffle_nodes', action='store_true', help='Shuffle nodes in the molecular graph')
    parser.add_argument('--save_image', action='store_true', help='Save image samples during training')

    # Inference parameters
    parser.add_argument('--beam_size', type=int, default=1, help='Beam size for inference')
    parser.add_argument('--n_best', type=int, default=1, help='Number of best results to return from beam search')
    parser.add_argument('--predict_coords', action='store_true', help='Predict atom coordinates in inference')
    parser.add_argument('--save_attns', action='store_true', help='Save attention weights')
    parser.add_argument('--molblock', action='store_true', help='Generate MOLBLOCK format output')
    parser.add_argument('--compute_confidence', action='store_true', help='Compute confidence scores')
    parser.add_argument('--keep_main_molecule', action='store_true', help='Keep the main molecule in post-processing')
    args = parser.parse_args()
    return args


def load_states(args, load_path):
    """
    Load model states (weights and optimizer states) from a given checkpoint path.
    """
    if load_path.endswith('.pth'):
        path = load_path
    elif args.load_ckpt == 'best':
        path = os.path.join(load_path, f'{args.decoder}_conv_best.pth')
    else:
        path = os.path.join(load_path, f'{args.decoder}_conv.pth')
    print_rank_0('Load ' + path)
    states = torch.load(path, map_location=torch.device('cpu'))
    return states


def safe_load(module, module_states):
    """
    Load model parameters with prefix removal for Distributed Data Parallel (DDP) compatibility.
    """
    def remove_prefix(state_dict):
        return {k.replace('module.', ''): v for k, v in state_dict.items()}

    missing_keys, unexpected_keys = module.load_state_dict(remove_prefix(module_states), strict=False)
    if missing_keys:
        print_rank_0('Missing keys: ' + str(missing_keys))
    if unexpected_keys:
        print_rank_0('Unexpected keys: ' + str(unexpected_keys))
    return


def get_model(args, tokenizer, device, load_path=None):
    """
    Initialize and return the encoder and decoder models, optionally loading from a checkpoint.
    """
    encoder = Encoder(args, pretrained=(not args.no_pretrained and load_path is None))
    args.encoder_dim = encoder.n_features

    decoder = Decoder(args, tokenizer)
    if load_path:
        states = load_states(args, load_path)
        safe_load(encoder, states['encoder'])
        safe_load(decoder, states['decoder'])
    encoder.to(device)
    decoder.to(device)

    if args.local_rank != -1:
        encoder = DDP(encoder, device_ids=[args.local_rank], output_device=args.local_rank)
        decoder = DDP(decoder, device_ids=[args.local_rank], output_device=args.local_rank)
        print_rank_0("DDP setup")

    return encoder, decoder


def get_optimizer_and_scheduler(args, encoder, decoder, load_path=None):
    """
    Initialize optimizers and schedulers for the encoder and decoder.
    """
    encoder_optimizer = AdamW(encoder.parameters(), lr=args.encoder_lr, weight_decay=args.weight_decay, amsgrad=False)
    encoder_scheduler = get_scheduler(args.scheduler, encoder_optimizer, args.num_warmup_steps, args.num_training_steps)

    decoder_optimizer = AdamW(decoder.parameters(), lr=args.decoder_lr, weight_decay=args.weight_decay, amsgrad=False)
    decoder_scheduler = get_scheduler(args.scheduler, decoder_optimizer, args.num_warmup_steps, args.num_training_steps)

    if load_path and args.resume:
        states = load_states(args, load_path)
        encoder_optimizer.load_state_dict(states['encoder_optimizer'])
        decoder_optimizer.load_state_dict(states['decoder_optimizer'])
        if args.init_scheduler:
            for group in encoder_optimizer.param_groups:
                group['lr'] = args.encoder_lr
            for group in decoder_optimizer.param_groups:
                group['lr'] = args.decoder_lr
        else:
            encoder_scheduler.load_state_dict(states['encoder_scheduler'])
            decoder_scheduler.load_state_dict(states['decoder_scheduler'])
        print_rank_0(f"Optimizer loaded from {load_path}")

    return encoder_optimizer, encoder_scheduler, decoder_optimizer, decoder_scheduler


def train_fn(train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch,
             encoder_scheduler, decoder_scheduler, scaler, device, global_step, SUMMARY, args):
    """
    Training loop for a single epoch, which includes forward and backward passes, gradient accumulation, and optimization.
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = LossMeter()
    
    # Switch to training mode
    encoder.train()
    decoder.train()
    
    start = end = time.time()
    encoder_grad_norm = decoder_grad_norm = 0

    for step, (indices, images, refs) in enumerate(train_loader):
        # Measure data loading time
        data_time.update(time.time() - end)
        images = images.to(device)
        batch_size = images.size(0)
        with torch.cuda.amp.autocast(enabled=args.fp16):
            features, hiddens = encoder(images, refs)
            results = decoder(features, hiddens, refs)
            losses = criterion(results, refs)
            loss = sum(losses.values())
        
        # Record loss
        loss_meter.update(loss, losses, batch_size)
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps
        scaler.scale(loss).backward()
        
        if (step + 1) % args.gradient_accumulation_steps == 0:
            scaler.unscale_(encoder_optimizer)
            scaler.unscale_(decoder_optimizer)
            encoder_grad_norm = torch.nn.utils.clip_grad_norm_(encoder.parameters(), args.max_grad_norm)
            decoder_grad_norm = torch.nn.utils.clip_grad_norm_(decoder.parameters(), args.max_grad_norm)
            scaler.step(encoder_optimizer)
            scaler.step(decoder_optimizer)
            scaler.update()
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            encoder_scheduler.step()
            decoder_scheduler.step()
            global_step += 1
        
        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if step % args.print_freq == 0 or step == (len(train_loader) - 1):
            loss_str = ' '.join([f'{k}:{v.avg:.4f}' for k, v in loss_meter.subs.items()])
            print_rank_0('Epoch: [{0}][{1}/{2}] '
                         'Running {remain:s} '
                         'Loss: {loss.avg:.4f} ({loss_str}) '
                         'Grad: {encoder_grad_norm:.4f}/{decoder_grad_norm:.4f} '
                         'lr: {encoder_lr:.5f} {decoder_lr:.5f}'
            .format(
                epoch + 1, step, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=loss_meter, loss_str=loss_str,
                sum_data_time=asMinutes(data_time.sum),
                remain=timeSince(start, float(step + 1) / len(train_loader)),
                encoder_grad_norm=encoder_grad_norm,
                decoder_grad_norm=decoder_grad_norm,
                encoder_lr=encoder_scheduler.get_lr()[0],
                decoder_lr=decoder_scheduler.get_lr()[0]))
            loss_meter.reset()
        if args.train_steps_per_epoch != -1 and (
                step + 1) // args.gradient_accumulation_steps == args.train_steps_per_epoch:
            break

    return loss_meter.epoch.avg, global_step


def valid_fn(valid_loader, encoder, decoder, tokenizer, device, args):
    """
    Validation function to evaluate model performance on validation data.
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    
    # Switch to evaluation mode
    encoder.eval()
    decoder.eval()
    predictions = {}
    start = end = time.time()
    
    # Perform inference on validation set
    for step, (indices, images, refs) in enumerate(valid_loader):
        data_time.update(time.time() - end)
        images = images.to(device)
        
        with torch.cuda.amp.autocast(enabled=args.fp16):
            with torch.no_grad():
                features, hiddens = encoder(images, refs)
                batch_preds = decoder.decode(features, hiddens, refs)
        
        for idx, preds in zip(indices, batch_preds):
            predictions[idx] = preds
        
        batch_time.update(time.time() - end)
        end = time.time()
        if step % args.print_freq == 0 or step == (len(valid_loader) - 1):
            print_rank_0('Evaluation: [{0}/{1}] '
                         'Spent {remain:s} '
            .format(
                step, len(valid_loader), batch_time=batch_time,
                data_time=data_time,
                sum_data_time=asMinutes(data_time.sum),
                remain=timeSince(start, float(step + 1) / len(valid_loader))))
    
    # Gather predictions from multiple GPUs
    gathered_preds = [None for i in range(dist.get_world_size())]
    dist.all_gather_object(gathered_preds, predictions)
    n = len(valid_loader.dataset)
    predictions = [{}] * n
    for preds in gathered_preds:
        for idx, pred in preds.items():
            predictions[idx] = pred
    return predictions


def train_loop(args, train_df, valid_df, aux_df, tokenizer, save_path):
    """
    Main training loop to iterate over epochs, perform training and validation, and save the model.
    """
    SUMMARY = None

    if args.local_rank == 0 and not args.debug:
        os.makedirs(save_path, exist_ok=True)
        save_args(args)
        SUMMARY = init_summary_writer(save_path)

    print_rank_0("Training started")
    device = args.device

    if aux_df is None:
        train_dataset = TrainDataset(args, train_df, tokenizer, split='train', dynamic_indigo=args.dynamic_indigo)
    else:
        train_dataset = AuxTrainDataset(args, train_df, aux_df, tokenizer)

    train_sampler = DistributedSampler(train_dataset, shuffle=True) if args.local_rank != -1 else RandomSampler(train_dataset)
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              sampler=train_sampler,
                              num_workers=args.num_workers,
                              prefetch_factor=4,
                              persistent_workers=True,
                              pin_memory=True,
                              drop_last=True,
                              collate_fn=bms_collate)

    if args.train_steps_per_epoch == -1:
        args.train_steps_per_epoch = len(train_loader) // args.gradient_accumulation_steps
    args.num_training_steps = args.epochs * args.train_steps_per_epoch
    args.num_warmup_steps = int(args.num_training_steps * args.warmup_ratio)

    # Initialize model and optimizer
    encoder, decoder = get_model(args, tokenizer, device, load_path=args.load_path)
    encoder_optimizer, encoder_scheduler, decoder_optimizer, decoder_scheduler = \
        get_optimizer_and_scheduler(args, encoder, decoder, load_path=args.load_path)
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)
    criterion = Criterion(args, tokenizer).to(device)

    best_score = -np.inf
    global_step = encoder_scheduler.last_epoch
    start_epoch = global_step // args.train_steps_per_epoch

    for epoch in range(start_epoch, args.epochs):
        if args.local_rank != -1:
            train_sampler.set_epoch(epoch)
            dist.barrier()

        start_time = time.time()
        avg_loss, global_step = train_fn(train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch,
                                         encoder_scheduler, decoder_scheduler, scaler, device, global_step, SUMMARY, args)
        
        scores = inference(args, valid_df, tokenizer, encoder, decoder, save_path, split='valid')

        if args.local_rank != 0:
            continue

        elapsed = time.time() - start_time
        print_rank_0(f'Epoch {epoch + 1} - Time: {elapsed:.0f}s')
        print_rank_0(f'Epoch {epoch + 1} - Score: ' + json.dumps(scores))

        save_obj = {
            'encoder': encoder.state_dict(),
            'encoder_optimizer': encoder_optimizer.state_dict(),
            'encoder_scheduler': encoder_scheduler.state_dict(),
            'decoder': decoder.state_dict(),
            'decoder_optimizer': decoder_optimizer.state_dict(),
            'decoder_scheduler': decoder_scheduler.state_dict(),
            'global_step': global_step,
            'args': {key: args.__dict__[key] for key in ['formats', 'input_size', 'coord_bins', 'sep_xy']}
        }

        for name in ['post_smiles', 'graph_smiles', 'canon_smiles']:
            if name in scores:
                score = scores[name]
                break

        if SUMMARY:
            SUMMARY.add_scalar('train/loss', avg_loss, global_step)
            encoder_lr = encoder_scheduler.get_lr()[0]
            decoder_lr = decoder_scheduler.get_lr()[0]
            SUMMARY.add_scalar('train/encoder_lr', encoder_lr, global_step)
            SUMMARY.add_scalar('train/decoder_lr', decoder_lr, global_step)
            for key in scores:
                SUMMARY.add_scalar(f'valid/{key}', scores[key], global_step)

        if score >= best_score:
            best_score = score
            print_rank_0(f'Epoch {epoch + 1} - Save Best Score: {best_score:.4f} Model')
            torch.save(save_obj, os.path.join(save_path, f'{args.encoder}_{args.decoder}_best.pth'))
            with open(os.path.join(save_path, 'best_valid.json'), 'w') as f:
                json.dump(scores, f)

        if args.save_mode == 'all':
            torch.save(save_obj, os.path.join(save_path, f'{args.encoder}_{args.decoder}_ep{epoch}.pth'))
        if args.save_mode == 'last':
            torch.save(save_obj, os.path.join(save_path, f'{args.encoder}_{args.decoder}_last.pth'))

    if args.local_rank != -1:
        dist.barrier()


def inference(args, data_df, tokenizer, encoder=None, decoder=None, save_path=None, split='test'):
    """
    Inference function to generate predictions on validation or test data.
    """
    print_rank_0("Inference started")
    device = args.device

    dataset = TrainDataset(args, data_df, tokenizer, split=split)
    sampler = DistributedSampler(dataset, shuffle=False) if args.local_rank != -1 else SequentialSampler(dataset)
    dataloader = DataLoader(dataset,
                            batch_size=args.batch_size * 2,
                            sampler=sampler,
                            num_workers=args.num_workers,
                            prefetch_factor=4,
                            persistent_workers=True,
                            pin_memory=True,
                            drop_last=False,
                            collate_fn=bms_collate)
    if encoder is None or decoder is None:
        encoder, decoder = get_model(args, tokenizer, device, args.load_path)
    predictions = valid_fn(dataloader, encoder, decoder, tokenizer, device, args)

    if args.local_rank != 0:
        return
    print('Start evaluation')

    if 'pubchem_cid' in data_df.columns:
        data_df['image_id'] = data_df['pubchem_cid']
    if 'image_id' not in data_df.columns:
        data_df['image_id'] = [path.split('/')[-1].split('.')[0] for path in data_df['file_path']]
    pred_df = data_df[['image_id']].copy()
    scores = {}

    for format_ in args.formats:
        if format_ in ['atomtok', 'atomtok_coords', 'chartok_coords']:
            format_preds = [preds[format_] for preds in predictions]
            pred_df['SMILES'] = [preds['smiles'] for preds in format_preds]
            if format_ in ['atomtok_coords', 'chartok_coords']:
                pred_df['node_coords'] = [preds['coords'] for preds in format_preds]
                pred_df['node_symbols'] = [preds['symbols'] for preds in format_preds]
            if args.compute_confidence:
                pred_df['SMILES_scores'] = [preds['scores'] for preds in format_preds]
                pred_df['indices'] = [preds['indices'] for preds in format_preds]

    if 'edges' in args.formats:
        pred_df['edges'] = [preds['edges'] for preds in predictions]
        if args.compute_confidence:
            pred_df['edges_scores'] = [preds['edges_scores'] for preds in predictions]
        smiles_list, molblock_list, r_success = convert_graph_to_smiles(
            pred_df['node_coords'], pred_df['node_symbols'], pred_df['edges'])
        pred_df['graph_SMILES'] = smiles_list
        if args.molblock:
            pred_df['molblock'] = molblock_list

    if 'SMILES' in pred_df.columns:
        if 'edges' in pred_df.columns:
            smiles_list, _, r_success = postprocess_smiles(
                pred_df['SMILES'], pred_df['node_coords'], pred_df['node_symbols'], pred_df['edges'])
        else:
            smiles_list, _, r_success = postprocess_smiles(pred_df['SMILES'])
        pred_df['post_SMILES'] = smiles_list

    if args.keep_main_molecule:
        if 'graph_SMILES' in pred_df:
            pred_df['graph_SMILES'] = keep_main_molecule(pred_df['graph_SMILES'])
        if 'post_SMILES' in pred_df:
            pred_df['post_SMILES'] = keep_main_molecule(pred_df['post_SMILES'])

    if 'SMILES' in data_df.columns:
        evaluator = SmilesEvaluator(data_df['SMILES'], tanimoto=True)
        if 'SMILES' in pred_df.columns:
            scores.update(evaluator.evaluate(pred_df['SMILES']))
        if 'post_SMILES' in pred_df.columns:
            post_scores = evaluator.evaluate(pred_df['post_SMILES'])
            scores['postprocessed_smiles'] = post_scores['canon_smiles']
            scores['postprocessed_graph_smiles'] = post_scores['graph']
            scores['postprocessed_chiral'] = post_scores['chiral']
            scores['postprocessed_tanimoto'] = post_scores['tanimoto']
        if 'graph_SMILES' in pred_df.columns:
            graph_scores = evaluator.evaluate(pred_df['graph_SMILES'])

    print('Saving predictions:')
    file = data_df.attrs['file'].split('/')[-1]
    pred_df = format_df(pred_df)
    if args.predict_coords:
        pred_df = pred_df[['image_id', 'SMILES', 'node_coords']]
    pred_df.to_csv(os.path.join(save_path, f'prediction_{file}'), index=False)

    if split == 'test':
        with open(os.path.join(save_path, f'eval_scores_{os.path.splitext(file)[0]}_{args.load_ckpt}.json'), 'w') as f:
            json.dump(scores, f)

    return scores


def get_chemdraw_data(args):
    """
    Load data for training, validation, and testing from specified file paths.
    """
    train_df, valid_df, test_df, aux_df = None, None, None, None
    if args.do_train:
        train_files = args.train_file.split(',')
        train_df = pd.concat([pd.read_csv(os.path.join(args.data_path, file)) for file in train_files])
        print_rank_0(f'train: {train_df.shape}')
        if args.aux_file:
            aux_df = pd.read_csv(os.path.join(args.data_path, args.aux_file))
            print_rank_0(f'aux: {aux_df.shape}')
    if args.do_train or args.do_valid:
        valid_df = pd.read_csv(os.path.join(args.data_path, args.valid_file))
        valid_df.attrs['file'] = args.valid_file
        print_rank_0(f'valid: {valid_df.shape}')
    if args.do_test:
        test_files = args.test_file.split(',')
        test_df = [pd.read_csv(os.path.join(args.data_path, file)) for file in test_files]
        for file, df in zip(test_files, test_df):
            df.attrs['file'] = file
            print_rank_0(file + f' test: {df.shape}')
    tokenizer = get_tokenizer(args)
    return train_df, valid_df, test_df, aux_df, tokenizer


def main():
    """
    Main function for model training, evaluation, and testing. Configures distributed training and orchestrates the workflow.
    """
    args = get_args()
    seed_torch(seed=args.seed)

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args.local_rank = int(os.environ['LOCAL_RANK'])
    if args.local_rank != -1:
        dist.init_process_group(backend=args.backend, init_method='env://', timeout=datetime.timedelta(0, 14400))
        torch.cuda.set_device(args.local_rank)
        torch.backends.cudnn.benchmark = True

    args.formats = args.formats.split(',')
    args.nodes = any([f in args.formats for f in ['atomtok_coords', 'chartok_coords']])
    args.edges = any([f in args.formats for f in ['atomtok_coords', 'chartok_coords']])

    train_df, valid_df, test_df, aux_df, tokenizer = get_chemdraw_data(args)

    if args.do_train:
        train_loop(args, train_df, valid_df, aux_df, tokenizer, args.save_path)

    if args.do_valid:
        scores = inference(args, valid_df, tokenizer, save_path=args.save_path, split='test')
        print_rank_0(json.dumps(scores, indent=4))

    if args.do_test:
        assert type(test_df) is list
        for df in test_df:
            scores = inference(args, df, tokenizer, save_path=args.save_path, split='test')
            print_rank_0(json.dumps(scores, indent=4))


if __name__ == "__main__":
    main()
