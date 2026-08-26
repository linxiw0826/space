import math
import sys

import torch

import utils
from models.sigreg import SIGReg


_sigreg = None


def get_sigreg(device, knots=17, num_proj=1024):
    global _sigreg
    if _sigreg is None:
        _sigreg = SIGReg(knots=knots, num_proj=num_proj).to(device)
    return _sigreg


def _unwrap(model):
    return model.module if hasattr(model, "module") else model


@torch.no_grad()
def update_router_bias(model):
    core = _unwrap(model)
    stats = getattr(core, "_route_stats", None) or []
    for block_idx, route_stats in stats:
        block = core.blocks[block_idx]
        select_frac = route_stats["select_frac"].detach().float()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(select_frac, op=torch.distributed.ReduceOp.SUM)
            select_frac /= torch.distributed.get_world_size()
        block.mlp.router.update_bias(select_frac)
        route_stats["expert_bias"] = block.mlp.router.expert_bias.detach().clone()


def _vec(tensor):
    return ",".join(f"{v:.3f}" for v in tensor.detach().float().cpu().tolist())


def print_route_stats(step, model):
    if not (step < 2 or step % 50 == 0):
        return
    core = _unwrap(model)
    for block_idx, stats in (getattr(core, "_route_stats", None) or []):
        print(
            f"[ROUTE step{step} block{block_idx}] "
            f"select_frac={_vec(stats['select_frac'])} "
            f"top1_frac={_vec(stats['top1_frac'])} "
            f"weight_frac={_vec(stats['weight_frac'])} "
            f"avg_selected_weight={_vec(stats['avg_selected_weight'])} "
            f"entropy={float(stats['router_entropy'].detach().float().cpu()):.4f} "
            f"expert_bias={','.join(f'{v:.4f}' for v in stats['expert_bias'].detach().float().cpu().tolist())}",
            flush=True,
        )


def train_one_epoch(model, data_loader, optimizer, device, epoch, loss_scaler,
                    max_norm=0, log_writer=None, start_steps=0,
                    lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, sigreg_weight=0.3):
    model.train()
    sigreg = get_sigreg(device)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("min_lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("future_loss", utils.SmoothedValue(window_size=20, fmt="{value:.4f}"))
    metric_logger.add_meter("sigreg_loss", utils.SmoothedValue(window_size=20, fmt="{value:.4f}"))

    header = f"Epoch: [{epoch}]"
    for step, batch in enumerate(metric_logger.log_every(data_loader, 1, header)):
        if num_training_steps_per_epoch is not None and step >= num_training_steps_per_epoch:
            break
        schedule_step = start_steps + step
        for group in optimizer.param_groups:
            if lr_schedule_values is not None:
                group["lr"] = lr_schedule_values[schedule_step] * group["lr_scale"]
            if wd_schedule_values is not None and group["weight_decay"] > 0:
                group["weight_decay"] = wd_schedule_values[schedule_step]

        images = batch[0].to(device, non_blocking=True)
        optimizer.zero_grad()

        with torch.amp.autocast("cuda"):
            future_loss, anchor_losses, anchors = model(images)
            core = _unwrap(model)
            with torch.amp.autocast("cuda", enabled=False):
                sigreg_loss = sigreg(core._last_x.float().transpose(0, 1))
            loss = future_loss + sigreg_weight * sigreg_loss

        grad_norm = loss_scaler(
            loss, optimizer, clip_grad=max_norm, parameters=model.parameters(),
            create_graph=False)
        loss_scale_value = loss_scaler.state_dict()["scale"]
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(2)

        torch.cuda.synchronize()
        update_router_bias(model)
        print_route_stats(step, model)

        lrs = [group["lr"] for group in optimizer.param_groups]
        wd = next((group["weight_decay"] for group in optimizer.param_groups if group["weight_decay"] > 0), None)
        metric_logger.update(
            loss=loss_value, loss_scale=loss_scale_value,
            future_loss=future_loss.item(), sigreg_loss=sigreg_loss.item(),
            lr=max(lrs), min_lr=min(lrs), grad_norm=grad_norm)
        if wd is not None:
            metric_logger.update(weight_decay=wd)

        if log_writer is not None:
            log_writer.update(loss=loss_value, future_loss=future_loss.item(),
                              sigreg_loss=sigreg_loss.item(), head="loss")
            for name, value in anchor_losses.items():
                log_writer.update(**{name: value.item()}, head="anchor")
            log_writer.update(lr=max(lrs), min_lr=min(lrs),
                              loss_scale=loss_scale_value, head="opt")
            log_writer.set_step()
        if step < 2 or step % 50 == 0:
            anchor_str = ",".join(str(int(a)) for a in anchors.detach().cpu().tolist())
            per_anchor = " ".join(f"{k}={v.item():.4f}" for k, v in anchor_losses.items())
            print(f"[ANCHOR step{step}] anchors={anchor_str} {per_anchor}", flush=True)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
