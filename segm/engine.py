import torch
import math

from segm.utils.logger import MetricLogger
from segm.metrics import gather_data, compute_metrics, calculate_cross_entropy_loss
from segm.model import utils
from segm.data.utils import IGNORE_LABEL
import segm.utils.torch as ptu
import wandb
import numpy as np




def train_one_epoch(
    model,
    data_loader,
    optimizer,
    lr_scheduler,
    epoch,
    amp_autocast,
    loss_scaler,
):
    criterion = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL)
    logger = MetricLogger(delimiter="  ")
    header = f"Epoch: [{epoch}]"
    print_freq = 500

    model.train()
    data_loader.set_epoch(epoch)
    num_updates = epoch * len(data_loader)
    for batch in logger.log_every(data_loader, print_freq, header):
        im = batch["im"].to(ptu.device)
        seg_gt = batch["segmentation"].long().to(ptu.device)
        text_inputs = batch["text"].to(ptu.device)

        with amp_autocast():
            seg_pred = model.forward(im, text_inputs)
            loss = criterion(seg_pred, seg_gt)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value), force=True)

        optimizer.zero_grad()
        if loss_scaler is not None:
            loss_scaler(
                loss,
                optimizer,
                parameters=model.parameters(),
            )
        else:
            loss.backward()
            optimizer.step()

        num_updates += 1
        lr_scheduler.step_update(num_updates=num_updates)

        torch.cuda.synchronize()

        logger.update(
            loss=loss.item(),
            learning_rate=optimizer.param_groups[0]["lr"],
        )

        wandb.log({'train loss': loss.item()})

    return logger


@torch.no_grad()
def evaluate(
    model,
    data_loader,
    val_seg_gt,
    window_size,
    window_stride,
    amp_autocast,
):

    model_without_ddp = model
    if hasattr(model, "module"):
        model_without_ddp = model.module
    logger = MetricLogger(delimiter="  ")
    header = "Eval:"
    print_freq = 50

    val_seg_pred = {}
    val_seg_prob = {}
    model.eval()
    for batch in logger.log_every(data_loader, print_freq, header):
        ims = [im.to(ptu.device) for im in batch["im"]]
        text_inputs =  batch["text"].to(ptu.device)
        ims_metas = batch["im_metas"]
        ori_shape = ims_metas[0]["ori_shape"]
        ori_shape = (ori_shape[0].item(), ori_shape[1].item())
        filename = batch["im_metas"][0]["ori_filename"][0]

        with amp_autocast():
            seg_pred = utils.inference(
                model_without_ddp,
                ims,
                text_inputs,
                ims_metas,
                ori_shape,
                window_size,
                window_stride,
                batch_size=1,
            )

            val_seg_prob[filename] = seg_pred
            seg_pred = seg_pred.argmax(0)
            
        seg_pred = seg_pred.cpu().numpy()
        val_seg_pred[filename] = seg_pred

    val_seg_pred = gather_data(val_seg_pred)
    val_seg_prob = gather_data(val_seg_prob)

    scores = compute_metrics(
        val_seg_pred,
        val_seg_gt,
        data_loader.unwrapped.n_cls,
        ignore_index=IGNORE_LABEL,
        distributed=ptu.distributed,
        ret_cat_iou = True
    )

    scores['ce_loss'] = calculate_cross_entropy_loss(val_seg_prob, val_seg_gt, ignore_index=IGNORE_LABEL, distributed = ptu.distributed)
    scores['mean_f1_score'] = np.round(np.nanmean(scores['f1score'].astype(np.float64)) * 100, 2)
    
    cls_names = ['farmland', 'water', 'forest','built-up', 'meadow']
    print("Class-wise F1 Scores:")
    for name, score in zip(cls_names, scores.pop('f1score')):
        print(f"  {name:<15}: {score:.2f}")
    
    print("Class-wise IOU Scores:")
    for name, score in zip(cls_names, scores.pop('cat_iou')):
        print(f"  {name:<15}: {score:.2f}")

    for k, v in scores.items():
        logger.update(**{f"{k}": v, "n": 1})
    
    wandb.log({
        "test loss": scores['ce_loss'],
        "mean acc": scores['mean_accuracy'],
        "mean iou": scores['mean_iou'],
        "mean f1": scores['mean_f1_score']

    })

    return logger
