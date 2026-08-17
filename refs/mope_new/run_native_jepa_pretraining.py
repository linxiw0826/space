import argparse
import datetime
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from timm.models import create_model

import models  # noqa: F401
import utils
from dataset.wisa_dataset import build_video_pretraining_dataset
from engine_native_jepa import train_one_epoch
from optim_factory import create_optimizer
from utils import NativeScalerWithGradNormCount as NativeScaler
from utils import multiple_pretrain_samples_collate


def get_args():
    parser = argparse.ArgumentParser("Native MoPE-JEPA pretraining")
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--save_ckpt_freq", default=10, type=int)
    parser.add_argument("--model", default="native_mope_jepa_base_patch16_224")
    parser.add_argument("--input_size", default=224, type=int)
    parser.add_argument("--num_frames", default=16, type=int)
    parser.add_argument("--tubelet_size", default=2, type=int)
    parser.add_argument("--sampling_rate", default=4, type=int)
    parser.add_argument("--num_sample", default=1, type=int)
    parser.add_argument("--datasets_root", default=os.environ.get("MOPE_DATA_LIST", ""))
    parser.add_argument("--general_data", default="")
    parser.add_argument("--general_max", default=0, type=int)
    parser.add_argument("--mask_type", default="tube", choices=["random", "tube", "random_temporal"])
    parser.add_argument("--mask_ratio", default=0.0, type=float)
    parser.add_argument("--drop_path", default=0.0, type=float)
    parser.add_argument("--predictor_dim", default=384, type=int)
    parser.add_argument("--predictor_depth", default=6, type=int)
    parser.add_argument("--predictor_num_heads", default=6, type=int)
    parser.add_argument("--pos_embed_type", default="fixed_1d", choices=["fixed_1d", "3d_sincos"])
    parser.add_argument("--predictor_pos_embed_type", default="learnable_1d", choices=["learnable_1d", "fixed_1d", "3d_sincos"])
    parser.add_argument("--encoder_depth", default=8, type=int)
    parser.add_argument("--dense_layers", default=4, type=int)
    parser.add_argument("--num_routed_experts", default=4, type=int)
    parser.add_argument("--candidate_k", default=2, type=int)
    parser.add_argument("--num_shared_experts", default=1, type=int)
    parser.add_argument("--router_score_func", default="sigmoid", choices=["sigmoid", "softmax"])
    parser.add_argument("--router_bias_update_speed", default=0.001, type=float)
    parser.add_argument("--future_num_anchors", default=3, type=int)
    parser.add_argument("--future_anchor_candidates", default="0,1,2,3,4,5,6")
    parser.add_argument("--future_anchor_weights", default="1.35,1.25,1.15,1.0,0.9,0.8,0.7")
    parser.add_argument("--sigreg_weight", default=0.3, type=float)
    parser.add_argument("--opt", default="adamw")
    parser.add_argument("--opt_eps", default=1e-8, type=float)
    parser.add_argument("--opt_betas", default=None, type=float, nargs="+")
    parser.add_argument("--clip_grad", default=1.0, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--weight_decay", default=0.05, type=float)
    parser.add_argument("--weight_decay_end", default=None, type=float)
    parser.add_argument("--lr", default=1.5e-4, type=float)
    parser.add_argument("--warmup_lr", default=1e-6, type=float)
    parser.add_argument("--min_lr", default=1e-5, type=float)
    parser.add_argument("--warmup_epochs", default=5, type=int)
    parser.add_argument("--warmup_steps", default=-1, type=int)
    parser.add_argument("--color_jitter", default=0.0, type=float)
    parser.add_argument("--train_interpolation", default="bicubic")
    parser.add_argument("--imagenet_default_mean_and_std", default=True, action="store_true")
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--log_dir", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="")
    parser.add_argument("--auto_resume", action="store_true")
    parser.add_argument("--no_auto_resume", action="store_false", dest="auto_resume")
    parser.set_defaults(auto_resume=True)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--max_train_steps_per_epoch", default=-1, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--pin_mem", action="store_true")
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)
    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument("--dist_url", default="env://")
    parser.add_argument("--data_path", default="")
    parser.add_argument("--data_root", default="")
    parser.add_argument("--fname_tmpl", default="img_{:05}.jpg")
    parser.add_argument("--decoder_mask_type", default="run_cell")
    parser.add_argument("--decoder_mask_ratio", default=0.0, type=float)
    return parser.parse_args()


def main(args):
    utils.init_distributed_mode(args)
    device = torch.device(args.device)
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    args.window_size = (
        args.num_frames // args.tubelet_size,
        args.input_size // 16,
        args.input_size // 16,
    )
    dataset_train = build_video_pretraining_dataset(args)
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=utils.get_world_size(),
        rank=utils.get_rank(), shuffle=True) if args.distributed else torch.utils.data.RandomSampler(dataset_train)
    log_writer = utils.TensorboardLogger(log_dir=args.log_dir) if args.log_dir and utils.is_main_process() else None
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=True,
        collate_fn=multiple_pretrain_samples_collate if args.num_sample > 1 else None)

    anchor_candidates = tuple(int(x) for x in args.future_anchor_candidates.split(",") if x.strip())
    anchor_weights = tuple(float(x) for x in args.future_anchor_weights.split(",") if x.strip())
    model = create_model(
        args.model, pretrained=False, drop_path_rate=args.drop_path,
        all_frames=args.num_frames, tubelet_size=args.tubelet_size,
        encoder_depth=args.encoder_depth, dense_layers=args.dense_layers,
        predictor_dim=args.predictor_dim, predictor_depth=args.predictor_depth,
        predictor_num_heads=args.predictor_num_heads,
        pos_embed_type=args.pos_embed_type,
        predictor_pos_embed_type=args.predictor_pos_embed_type,
        num_experts=args.num_routed_experts, top_k=args.candidate_k,
        num_shared_experts=args.num_shared_experts,
        router_score_func=args.router_score_func,
        router_bias_update_speed=args.router_bias_update_speed,
        num_anchors=args.future_num_anchors,
        anchor_candidates=anchor_candidates,
        anchor_weights=anchor_weights)
    if torch.__version__ >= "2":
        torch.set_float32_matmul_precision("high")
    model.to(device)
    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_batch_size = args.batch_size * utils.get_world_size()
    num_steps = len(data_loader_train)
    if args.max_train_steps_per_epoch and args.max_train_steps_per_epoch > 0:
        num_steps = min(num_steps, args.max_train_steps_per_epoch)
    args.lr = args.lr * total_batch_size / 256
    args.min_lr = args.min_lr * total_batch_size / 256
    args.warmup_lr = args.warmup_lr * total_batch_size / 256
    print(f"Model params: {n_parameters / 1e6:.2f}M")
    print(f"LR={args.lr:.8f} total_batch_size={total_batch_size} steps/epoch={num_steps}")
    print(f"Anchors candidates={anchor_candidates} weights={anchor_weights}")

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu],
            find_unused_parameters=False, static_graph=True)
        model_without_ddp = model.module

    optimizer = create_optimizer(args, model_without_ddp)
    loss_scaler = NativeScaler()
    lr_schedule = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_steps,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps)
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, num_steps)
    utils.auto_load_model(args=args, model=model, model_without_ddp=model_without_ddp,
                          optimizer=optimizer, loss_scaler=loss_scaler)

    start = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        if log_writer is not None:
            log_writer.set_step(epoch * num_steps)
        stats = train_one_epoch(
            model, data_loader_train, optimizer, device, epoch, loss_scaler,
            args.clip_grad, log_writer=log_writer,
            start_steps=epoch * num_steps, lr_schedule_values=lr_schedule,
            wd_schedule_values=wd_schedule, num_training_steps_per_epoch=num_steps,
            sigreg_weight=args.sigreg_weight)
        if args.output_dir:
            save_epoch = epoch + 1
            if save_epoch % args.save_ckpt_freq == 0 or save_epoch == args.epochs:
                utils.save_model(args=args, model=model, model_without_ddp=model_without_ddp,
                                 optimizer=optimizer, loss_scaler=loss_scaler,
                                 epoch=save_epoch)
        if args.output_dir and utils.is_main_process():
            with open(os.path.join(args.output_dir, "log.txt"), "a", encoding="utf-8") as f:
                f.write(json.dumps({**{f"train_{k}": v for k, v in stats.items()},
                                    "epoch": epoch, "n_parameters": n_parameters}) + "\n")
    print("Training time", datetime.timedelta(seconds=int(time.time() - start)))


if __name__ == "__main__":
    try:
        torch.multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    opts = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    if opts.log_dir:
        Path(opts.log_dir).mkdir(parents=True, exist_ok=True)
    main(opts)
