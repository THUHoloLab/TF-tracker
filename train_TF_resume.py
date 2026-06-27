import logging
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf

from utils.callbacks import IncreaseSequenceLengthCallback
from utils.utils import *

logger = logging.getLogger(__name__)

torch.set_num_threads(1)
torch.backends.cudnn.benchmark = True


def get_cfg_str(cfg, key, default="none"):
    """
    Safely get string config value.
    This avoids errors if resume_path is not defined in yaml.
    """
    value = OmegaConf.select(cfg, key, default=default)
    if value is None:
        value = default
    return str(value)

def freeze_selected_modules(model):
    """
    Freeze specified modules and only train the remaining modules.
    """

    freeze_module_names = [
        "reference_encoder",
        "target_encoder",
    ]

    for module_name in freeze_module_names:
        if not hasattr(model, module_name):
            logger.warning(f"[Freeze] Model has no module named: {module_name}")
            continue

        module = getattr(model, module_name)

        for param in module.parameters():
            param.requires_grad = False

        module.eval()

        logger.info(f"[Freeze] Frozen module: {module_name}")

    # 统计参数量
    total_params = 0
    trainable_params = 0
    frozen_params = 0

    for name, param in model.named_parameters():
        numel = param.numel()
        total_params += numel

        if param.requires_grad:
            trainable_params += numel
        else:
            frozen_params += numel

    logger.info(f"[Freeze] Total params: {total_params / 1e6:.3f} M")
    logger.info(f"[Freeze] Trainable params: {trainable_params / 1e6:.3f} M")
    logger.info(f"[Freeze] Frozen params: {frozen_params / 1e6:.3f} M")

    logger.info("[Freeze] Trainable parameter names:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.info(f"  [Trainable] {name}")

    return model


@hydra.main(
    version_base=None,
    config_path="configs",
    config_name="train_defaults"
)
def train(cfg):
    pl.seed_everything(1234)

    # Update configuration dicts with common keys
    propagate_keys(cfg)
    logger.info("\n" + OmegaConf.to_yaml(cfg))

    # =========================================================
    # Resume path: 用于中断后完整恢复训练
    # checkpoint_path: 用于加载预训练权重 / fine-tuning
    # =========================================================
    resume_path = get_cfg_str(cfg, "resume_path", default="none")
    checkpoint_path = get_cfg_str(cfg, "checkpoint_path", default="none")

    is_resume = resume_path.lower() != "none"
    is_load_weight = checkpoint_path.lower() != "none"

    # =========================================================
    # Instantiate model
    # =========================================================
    model = hydra.utils.instantiate(
        cfg.model,
        _recursive_=False,
    )

    # =========================================================
    # 如果是 resume，中断续训，不要手动 load_from_checkpoint
    # Lightning 会在 trainer.fit(..., ckpt_path=resume_path) 中恢复：
    # global_step / optimizer / lr_scheduler / trainer state
    #
    # 如果不是 resume，但 checkpoint_path != none，则表示只加载权重，
    # 适合 fine-tuning 或重新开始训练。
    # =========================================================
    if is_resume:
        logger.info(f"Resume training from checkpoint: {resume_path}")
        logger.info("Do not call model.load_from_checkpoint when resuming.")
    elif is_load_weight:
        logger.info(f"Load model weights from checkpoint: {checkpoint_path}")

        model = model.load_from_checkpoint(
            checkpoint_path=checkpoint_path
        )

        # Override stuff for fine-tuning / loading pretrained weights
        model.hparams.optimizer.lr = cfg.model.optimizer.lr
        model.hparams.optimizer._target_ = cfg.model.optimizer._target_
        model.debug = True
        model.unrolls = cfg.init_unrolls
        model.max_unrolls = cfg.max_unrolls
        model.pose_mode = cfg.model.pose_mode
        model = freeze_selected_modules(model)

    # =========================================================
    # Instantiate datamodule
    # =========================================================
    data_module = hydra.utils.instantiate(cfg.data)

    # =========================================================
    # Logging
    # =========================================================
    if cfg.logging:
        training_logger = pl.loggers.TensorBoardLogger(
            ".", "", "", log_graph=True, default_hp_metric=False
        )
    else:
        training_logger = None

    # =========================================================
    # Checkpoint callback
    # 建议保留 save_last=True，方便下次直接从 last.ckpt 续训
    # =========================================================
    ckpt_5epoch = pl.callbacks.ModelCheckpoint(
        save_top_k=-1,
        every_n_epochs=10,
        save_last=True,
        filename="epoch={epoch:02d}-step={step}",
    )

    ckpt_5epoch._enable_version_counter = False

    callbacks = [
        IncreaseSequenceLengthCallback(
            unroll_factor=cfg.unroll_factor,
            schedule=cfg.unroll_schedule,
        ),
        pl.callbacks.LearningRateMonitor(logging_interval="epoch"),
        ckpt_5epoch,
    ]

    # =========================================================
    # Trainer
    # =========================================================
    '''trainer = pl.Trainer(
        **OmegaConf.to_container(cfg.trainer),
        accelerator="gpu",
        devices=3,
        strategy="ddp",
        callbacks=callbacks,
        logger=training_logger,
    )'''
    trainer = pl.Trainer(
        **OmegaConf.to_container(cfg.trainer),
        devices=[0],
        accelerator="gpu",
        callbacks=callbacks,
        logger=training_logger
    )
    # =========================================================
    # Fit
    # 关键：resume 时通过 ckpt_path 恢复完整训练状态
    # =========================================================
    resume_ckpt = resume_path if is_resume else None

    trainer.fit(
        model,
        datamodule=data_module,
        ckpt_path=resume_ckpt,
    )


if __name__ == "__main__":
    train()