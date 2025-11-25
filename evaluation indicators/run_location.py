# coding=utf-8
# @name:        run_location.py
# @software:    PyCharm
# @description: Hydraulic network burst point localization system

import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from model.location.RNN_model import MyRNN
from model.location.CNN_model import MyCNN
from model.location.FADenseNet import FADenseNet
from datas.dataset import MyLocationData
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
import config  # Import configuration module

# Config-driven initialization
torch.manual_seed(config.SEED)
torch.cuda.manual_seed(config.SEED)
device = config.DEVICE


def train_pipeline(model, train_loader, val_loaders):
    """
    Unified training workflow with multi-validation tracking
    Args:
        model: Initialized torch.nn.Module
        train_loader: DataLoader for training set
        val_loaders: List of DataLoaders for validation sets
    Returns:
        np.array: Metrics matrix [epoch x metrics]
    """
    # Initialize training components
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    class_weights = torch.tensor(config.CLASS_WEIGHTS, dtype=config.DTYPE, device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    trainer = create_supervised_trainer(model, optimizer, loss_fn, device=device)
    metrics = {
        "accuracy": Accuracy(device=device),
        "loss": Loss(loss_fn, device=device)
    }
    evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)

    # Prepare results container [epoch, train_acc, train_loss, val1_acc, val1_loss,...]
    results_cols = 2 + 2 * len(val_loaders)
    log_matrix = np.zeros((config.MAX_EPOCHS, results_cols))

    @trainer.on(Events.EPOCH_COMPLETED)
    def evaluate_all(trainer):
        epoch = trainer.state.epoch - 1

        # Evaluate training set
        evaluator.run(train_loader)
        train_metrics = evaluator.state.metrics
        log_matrix[epoch, 0] = train_metrics["accuracy"]
        log_matrix[epoch, 1] = train_metrics["loss"]

        # Evaluate each validation set
        for idx, val_loader in enumerate(val_loaders):
            evaluator.run(val_loader)
            val_metrics = evaluator.state.metrics
            log_matrix[epoch, 2 + 2 * idx] = val_metrics["accuracy"]
            log_matrix[epoch, 3 + 2 * idx] = val_metrics["loss"]

    trainer.run(train_loader, max_epochs=config.MAX_EPOCHS)
    return log_matrix


def initialize_model(model_type):
    """Model factory based on selected architecture"""
    if model_type == "RNN":
        return MyRNN(
            config.INPUT_FEATURES,
            config.GRU_HIDDEN,
            config.LINEAR_HIDDEN,
            config.OUTPUT_CLASSES
        ).to(device)
    elif model_type == "CNN":
        return MyCNN(
            config.IN_CHANNEL,
            config.HIDDEN_CHANNEL1,
            config.HIDDEN_CHANNEL2,
            config.OUT_CHANNEL,
            config.HIDDEN_FEATURE,
            config.OUTPUT_CLASSES
        ).to(device)
    elif model_type == "FA":
        return FADenseNet(
            config.INPUT_FEATURES,
            config.OUTPUT_CLASSES,
            config.SERIOUS_LEN
        ).to(device)


def get_data_loaders():
    """Prepare data loaders for training and validation"""
    train_dataset = MyLocationData(
        config.TRAIN_DATA_PATH,
        config.TRAIN_LABEL_PATH,
        need_time_len=config.SERIOUS_LEN,
        left_offset=config.TIME_OFFSETS[0],
        right_offset=config.TIME_OFFSETS[1]
    )

    # Validation datasets for different pipe diameters
    val_datasets = [
        MyLocationData(p_path, l_path, config.SERIOUS_LEN)
        for p_path, l_path in zip(
            config.VAL_PRESSURE_PATHS,
            config.VAL_LABEL_PATHS
        )
    ]

    train_loader = DataLoader(train_dataset, config.BATCH_SIZE, shuffle=True)
    val_loaders = [
        DataLoader(ds, config.BATCH_SIZE, shuffle=False)
        for ds in val_datasets
    ]

    return train_loader, val_loaders


def main():
    """Centralized execution workflow"""
    # Prepare data
    train_loader, val_loaders = get_data_loaders()

    # Initialize model
    model = initialize_model(config.MODEL_TYPE)

    # Execute training
    metrics = train_pipeline(model, train_loader, val_loaders)

    # Save outputs
    if config.SAVE_MODEL:
        torch.save(model.state_dict(), config.MODEL_SAVE_PATH)

    if config.SAVE_RESULTS:
        np.savetxt(
            config.RESULTS_PATH,
            metrics,
            fmt='%.4f',
            delimiter=',',
            header=','.join(
                'train_acc', 'train_loss',
                *[f'val{i + 1}_acc,val{i + 1}_loss' for i in range(len(val_loaders))]
            )
        )

        if __name__ == '__main__':
            main()
