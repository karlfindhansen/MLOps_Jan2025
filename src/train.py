import os
import shutil
import dotenv
import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from hydra_loggers import HydraRichLogger, get_hydra_dir_and_job_name
from model import CustomClassifier
import torch

# Optional: Remove WandB import as it's no longer needed
# import wandb

dotenv.load_dotenv()
logger = HydraRichLogger(level=os.getenv("LOG_LEVEL", "INFO"))

def train_model(cfg: DictConfig, train_dataloader, val_dataloader, test_dataloader):
    """Train and evaluate the model."""
    logger.info("Starting training script")
    
    # Use a fallback for logdir if it's not in the config
    logdir = cfg.get("logdir", os.path.join(os.getcwd(), "logs"))
    logger.info(f"Logging to {logdir}")
    os.makedirs(f"{logdir}/checkpoints", exist_ok=True)

    # Instantiate model and datamodule
    logger.info("Initializing model and datamodule")
    model = CustomClassifier(
        backbone=cfg.model.backbone,
        num_classes=cfg.model.num_classes,
        pretrained=cfg.model.pretrained,
        learning_rate=cfg.model.learning_rate,
        optimizer=cfg.model.optimizer,
    )
    device = torch.device('cpu')  # Use CPU instead of MPS
    model.to(device)

    # Instantiate logger and callbacks
    experiment_logger = hydra.utils.instantiate(cfg.experiment_logger, save_dir=logdir)
    experiment_logger.log_hyperparams(OmegaConf.to_container(cfg))

    early_stopping_callback = hydra.utils.instantiate(cfg.callbacks.early_stopping)
    checkpoint_callback = hydra.utils.instantiate(cfg.callbacks.checkpoint, dirpath=f"{logdir}/checkpoints")
    learning_rate_callback = hydra.utils.instantiate(cfg.callbacks.learning_rate_monitor)
    progress_bar_callback = hydra.utils.instantiate(cfg.callbacks.progress_bar)

    # Instantiate trainer
    logger.info("Setting up trainer")
    trainer = pl.Trainer(
        default_root_dir=logdir,
        logger=experiment_logger,
        callbacks=[checkpoint_callback, early_stopping_callback, learning_rate_callback, progress_bar_callback],
        **cfg.trainer,
    )

    if cfg.train:
        logger.info("Starting training")
        trainer.fit(model, train_dataloader, val_dataloader)

    if cfg.evaluate:
        logger.info("Starting evaluation")
        results = trainer.test(model, test_dataloader)

    if cfg.upload_model:
        logger.info("Saving model as artifact")
        best_model = checkpoint_callback.best_model_path
        os.makedirs(f"{logdir}/checkpoints", exist_ok=True)
        shutil.copy(best_model, f"{logdir}/checkpoints/checkpoint.ckpt")
        # Skip the WandB artifact upload and just save the model locally
        # artifact = wandb.Artifact(
        #     name=cfg.model.artifact_name,
        #     type="model",
        #     metadata={k.lstrip("test_"): round(v, 3) for k, v in results[0].items()},
        # )
        # artifact.add_file(f"{logdir}/checkpoints/checkpoint.ckpt")
        # experiment_logger.experiment.log_artifact(artifact)
        logger.info(f"Model saved at {logdir}/checkpoints/checkpoint.ckpt")

    return best_model
