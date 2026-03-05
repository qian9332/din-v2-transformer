#!/usr/bin/env python3
"""
Training script for DIN-V1 and DIN-V2 models.
Supports GPU training, logging, checkpointing, and TensorBoard.
"""

import os
import sys
import time
import json
import argparse
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import get_dataloaders, NUM_BEHAVIOR_TYPES
from src.model import DINV2
from src.model_v1 import DINV1
from src.utils import (
    setup_logger, calculate_metrics, EarlyStopping,
    AverageMeter, count_parameters, format_time
)


def parse_args():
    parser = argparse.ArgumentParser(description='Train DIN-V1/V2 model')
    parser.add_argument('--model', type=str, default='v2', choices=['v1', 'v2'])
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_transformer_layers', type=int, default=2)
    parser.add_argument('--max_seq_len', type=int, default=50)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--data_path', type=str, default='data/UserBehavior.csv')
    parser.add_argument('--output_dir', type=str, default='data/')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--min_hist_len', type=int, default=5)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--log_dir', type=str, default='logs/')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/')
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--eval_interval', type=int, default=5000)
    return parser.parse_args()


def get_device(device_str):
    if device_str == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device('cpu')
            print("GPU not available, using CPU")
    else:
        device = torch.device(device_str)
    return device


def train_epoch(model, train_loader, optimizer, criterion, device,
                epoch, args, logger, writer, global_step, val_loader=None):
    model.train()
    loss_meter = AverageMeter()
    batch_time = AverageMeter()
    epoch_labels = []
    epoch_preds = []
    start_time = time.time()

    for batch_idx, batch in enumerate(train_loader):
        batch_start = time.time()
        hist_items = batch['hist_items'].to(device)
        hist_behaviors = batch['hist_behaviors'].to(device)
        hist_categories = batch['hist_categories'].to(device)
        mask = batch['mask'].to(device)
        target_item = batch['target_item'].to(device)
        target_category = batch['target_category'].to(device)
        labels = batch['label'].to(device)

        logits = model(hist_items, hist_behaviors, hist_categories, mask,
                      target_item, target_category)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        loss_meter.update(loss.item(), hist_items.size(0))
        batch_time.update(time.time() - batch_start)

        preds = torch.sigmoid(logits).detach().cpu().numpy()
        epoch_labels.extend(labels.cpu().numpy().flatten().tolist())
        epoch_preds.extend(preds.flatten().tolist())
        global_step += 1

        if (batch_idx + 1) % args.log_interval == 0:
            elapsed = time.time() - start_time
            samples_per_sec = (batch_idx + 1) * args.batch_size / elapsed
            msg = (f"Epoch [{epoch+1}/{args.epochs}] "
                   f"Batch [{batch_idx+1}/{len(train_loader)}] "
                   f"Loss: {loss_meter.avg:.4f} "
                   f"Speed: {samples_per_sec:.0f} samples/s "
                   f"ETA: {format_time((len(train_loader) - batch_idx - 1) * batch_time.avg)}")
            logger.info(msg)
            writer.add_scalar('train/loss', loss_meter.val, global_step)
            writer.add_scalar('train/loss_avg', loss_meter.avg, global_step)
            writer.add_scalar('train/speed', samples_per_sec, global_step)

        if val_loader is not None and (batch_idx + 1) % args.eval_interval == 0:
            val_metrics = evaluate(model, val_loader, criterion, device)
            logger.info(f"  [Mid-Epoch Eval] AUC: {val_metrics['auc']:.4f} "
                       f"LogLoss: {val_metrics['logloss']:.4f} "
                       f"Acc: {val_metrics['accuracy']:.4f}")
            writer.add_scalar('val/auc_step', val_metrics['auc'], global_step)
            model.train()

    epoch_metrics = calculate_metrics(np.array(epoch_labels), np.array(epoch_preds))
    epoch_metrics['loss'] = loss_meter.avg
    return epoch_metrics, global_step


def evaluate(model, data_loader, criterion, device):
    model.eval()
    all_labels = []
    all_preds = []
    total_loss = 0
    total_samples = 0

    with torch.no_grad():
        for batch in data_loader:
            hist_items = batch['hist_items'].to(device)
            hist_behaviors = batch['hist_behaviors'].to(device)
            hist_categories = batch['hist_categories'].to(device)
            mask = batch['mask'].to(device)
            target_item = batch['target_item'].to(device)
            target_category = batch['target_category'].to(device)
            labels = batch['label'].to(device)

            logits = model(hist_items, hist_behaviors, hist_categories, mask,
                          target_item, target_category)
            loss = criterion(logits, labels)

            preds = torch.sigmoid(logits).cpu().numpy()
            all_labels.extend(labels.cpu().numpy().flatten().tolist())
            all_preds.extend(preds.flatten().tolist())
            total_loss += loss.item() * hist_items.size(0)
            total_samples += hist_items.size(0)

    metrics = calculate_metrics(np.array(all_labels), np.array(all_preds))
    metrics['loss'] = total_loss / max(total_samples, 1)
    return metrics


def main():
    args = parse_args()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_tag = f"din_{args.model}_{timestamp}"

    log_dir = os.path.join(args.log_dir, model_tag)
    checkpoint_dir = os.path.join(args.checkpoint_dir, model_tag)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    logger = setup_logger(log_dir, name=model_tag)
    logger.info("=" * 80)
    logger.info(f"DIN-{args.model.upper()} Training Started")
    logger.info("=" * 80)
    logger.info(f"Arguments: {json.dumps(vars(args), indent=2)}")

    writer = SummaryWriter(log_dir=log_dir)
    device = get_device(args.device)
    logger.info(f"Device: {device}")

    logger.info("\n--- Loading Data ---")
    feature_dims, train_loader, val_loader, test_loader = get_dataloaders(
        data_path=args.data_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        num_workers=args.num_workers,
        min_hist_len=args.min_hist_len,
    )
    logger.info(f"Feature dimensions: {feature_dims}")

    logger.info("\n--- Building Model ---")
    if args.model == 'v2':
        model = DINV2(
            num_items=feature_dims['num_items'],
            num_categories=feature_dims['num_categories'],
            num_behaviors=NUM_BEHAVIOR_TYPES,
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            num_transformer_layers=args.num_transformer_layers,
            max_seq_len=args.max_seq_len,
            dropout=args.dropout,
        )
    else:
        model = DINV1(
            num_items=feature_dims['num_items'],
            num_categories=feature_dims['num_categories'],
            embed_dim=args.embed_dim,
            dropout=args.dropout,
        )

    model = model.to(device)
    total_params, trainable_params = count_parameters(model)
    logger.info(f"Model: {model.model_name}")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Model architecture:\n{model}")

    config = {
        'model_version': args.model,
        'feature_dims': feature_dims,
        'args': vars(args),
        'total_params': total_params,
        'trainable_params': trainable_params,
    }
    with open(os.path.join(log_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=1, 
    )
    early_stopping = EarlyStopping(patience=args.patience, mode='max', )

    logger.info("\n--- Training ---")
    global_step = 0
    best_val_auc = 0.0
    training_history = []

    for epoch in range(args.epochs):
        epoch_start = time.time()
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch+1}/{args.epochs}")
        logger.info(f"{'='*60}")

        train_metrics, global_step = train_epoch(
            model, train_loader, optimizer, criterion, device,
            epoch, args, logger, writer, global_step, val_loader
        )

        train_time = time.time() - epoch_start
        logger.info(f"\n[Train] Epoch {epoch+1} completed in {format_time(train_time)}")
        logger.info(f"  Loss: {train_metrics['loss']:.4f}")
        logger.info(f"  AUC: {train_metrics['auc']:.4f}")
        logger.info(f"  Accuracy: {train_metrics['accuracy']:.4f}")

        logger.info("\nEvaluating on validation set...")
        val_start = time.time()
        val_metrics = evaluate(model, val_loader, criterion, device)
        val_time = time.time() - val_start

        logger.info(f"[Val] Evaluation completed in {format_time(val_time)}")
        logger.info(f"  Loss: {val_metrics['loss']:.4f}")
        logger.info(f"  AUC: {val_metrics['auc']:.4f}")
        logger.info(f"  LogLoss: {val_metrics['logloss']:.4f}")
        logger.info(f"  Accuracy: {val_metrics['accuracy']:.4f}")

        writer.add_scalar('train/epoch_loss', train_metrics['loss'], epoch)
        writer.add_scalar('train/epoch_auc', train_metrics['auc'], epoch)
        writer.add_scalar('val/loss', val_metrics['loss'], epoch)
        writer.add_scalar('val/auc', val_metrics['auc'], epoch)
        writer.add_scalar('val/logloss', val_metrics['logloss'], epoch)
        writer.add_scalar('val/accuracy', val_metrics['accuracy'], epoch)

        scheduler.step(val_metrics['auc'])

        epoch_history = {
            'epoch': epoch + 1,
            'train_loss': train_metrics['loss'],
            'train_auc': train_metrics['auc'],
            'val_loss': val_metrics['loss'],
            'val_auc': val_metrics['auc'],
            'val_logloss': val_metrics['logloss'],
            'val_accuracy': val_metrics['accuracy'],
            'train_time': train_time,
            'val_time': val_time,
            'lr': optimizer.param_groups[0]['lr'],
        }
        training_history.append(epoch_history)

        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc': val_metrics['auc'],
                'val_loss': val_metrics['loss'],
                'config': config,
            }, checkpoint_path)
            logger.info(f"  * New best model saved! Val AUC: {val_metrics['auc']:.4f}")

        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_auc': val_metrics['auc'],
            'config': config,
        }, os.path.join(checkpoint_dir, 'latest_model.pt'))

        early_stopping(val_metrics['auc'], model)
        if early_stopping.early_stop:
            logger.info("Early stopping triggered!")
            break

    # Test
    logger.info("\n--- Testing ---")
    best_ckpt = os.path.join(checkpoint_dir, 'best_model.pt')
    if os.path.exists(best_ckpt):
        checkpoint = torch.load(best_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded best model from epoch {checkpoint['epoch']}")

    test_metrics = evaluate(model, test_loader, criterion, device)
    logger.info(f"[Test] AUC: {test_metrics['auc']:.4f}")
    logger.info(f"  LogLoss: {test_metrics['logloss']:.4f}")
    logger.info(f"  Accuracy: {test_metrics['accuracy']:.4f}")

    writer.add_scalar('test/auc', test_metrics['auc'], 0)
    writer.add_scalar('test/logloss', test_metrics['logloss'], 0)

    results = {
        'model': args.model,
        'model_name': model.model_name,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'best_val_auc': best_val_auc,
        'test_metrics': test_metrics,
        'training_history': training_history,
        'feature_dims': feature_dims,
        'args': vars(args),
    }
    results_path = os.path.join(log_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"\n{'='*80}")
    logger.info(f"Training Complete!")
    logger.info(f"  Model: {model.model_name}")
    logger.info(f"  Best Val AUC: {best_val_auc:.4f}")
    logger.info(f"  Test AUC: {test_metrics['auc']:.4f}")
    logger.info(f"  Results saved to: {results_path}")
    logger.info(f"{'='*80}")

    writer.close()
    return test_metrics


if __name__ == '__main__':
    main()
