# coding=utf-8
# @name:        run_improve_location.py
# @software:    PyCharm
# @description: Multi-source data fusion pipeline for pipe burst localization

import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from model.improve_location.IRNN import IRNN
from model.improve_location.ICNN import ICNN
from model.improve_location.IFA import IFA
from datas.dataset import MyLocationDataQ
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, ConfusionMatrix, ClassificationReport
import config  # Import configuration settings

# Configuration-based setup
device = config.DEVICE
torch.manual_seed(config.SEED)
torch.cuda.manual_seed(config.SEED) if torch.cuda.is_available() else None


def initialize_model(model_type):
    """Initialize model based on selected type"""
    if model_type == "IRNN":
        return IRNN(
            config.SENSOR_GRAPH,
            config.INPUT_FEATURES,
            config.GRU_HIDDEN_SIZE,
            config.LINEAR_HIDDEN,
            config.FLOW_FUSION_DIM,
            config.NUM_CLASSES
        ).to(device)
    elif model_type == "ICNN":
        return ICNN(
            config.SENSOR_GRAPH,
            config.NUM_PRESSURE_SENSORS,
            config.INPUT_CHANNELS,
            config.CONV_CHANNELS[0],
            config.CONV_CHANNELS[1],
            config.OUTPUT_CHANNELS,
            config.LATENT_DIMENSION,
            config.NUM_CLASSES
        ).to(device)
    elif model_type == "IFA":
        return IFA(
            config.SENSOR_GRAPH,
            config.NUM_PRESSURE_SENSORS,
            config.INPUT_FEATURES,
            config.NUM_CLASSES,
            config.SEQUENCE_LENGTH
        ).to(device)


def create_data_loaders():
    """Prepare data loaders for training and validation"""
    train_dataset = MyLocationDataQ(
        config.TRAIN_PRESSURE_PATH,
        config.TRAIN_LABEL_PATH,
        config.TRAIN_FLOW_PATH,
        need_time_len=config.SEQUENCE_LENGTH,
        left_offset=config.TIME_OFFSETS[0],
        right_offset=config.TIME_OFFSETS[1]
    )

    # Validation datasets
    val_datasets = [
        MyLocationDataQ(p_path, l_path, q_path, config.SEQUENCE_LENGTH)
        for p_path, l_path, q_path in zip(
            config.VAL_PRESSURE_PATHS,
            config.VAL_LABEL_PATHS,
            config.VAL_FLOW_PATHS
        )
    ]

    train_loader = DataLoader(train_dataset, config.BATCH_SIZE, shuffle=True)
    val_loaders = [
        DataLoader(ds, config.BATCH_SIZE, shuffle=False)
        for ds in val_datasets
    ]

    return train_loader, val_loaders


def train_pipeline(model):
    """Full training workflow with validation tracking"""
    # Initialize components
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    class_weights = torch.tensor(config.CLASS_WEIGHTS, dtype=torch.float32, device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    # Create engines
    trainer = create_supervised_trainer(model, optimizer, loss_fn, device=device)
    val_metrics = {
        "accuracy": Accuracy(device=device),
        "loss": Loss(loss_fn, device=device)
    }
    evaluator = create_supervised_evaluator(model, metrics=val_metrics, device=device)

    # Results container [train_acc, train_loss, val1_acc, val1_loss, ...]
    log_matrix = np.zeros((config.MAX_EPOCHS, 2 * (len(config.VAL_PRESSURE_PATHS) + 1)))

    @trainer.on(Events.EPOCH_COMPLETED)
    def evaluate_model(engine):
        epoch = engine.state.epoch - 1
        # Training metrics
        evaluator.run(train_loader)
        train_metrics = evaluator.state.metrics
        log_matrix[epoch, 0] = train_metrics["accuracy"]
        log_matrix[epoch, 1] = train_metrics["loss"]

        # Validation metrics
        for idx, loader in enumerate(val_loaders):
            evaluator.run(loader)
            val_metrics = evaluator.state.metrics
            log_matrix[epoch, 2 * idx + 2] = val_metrics["accuracy"]
            log_matrix[epoch, 2 * idx + 3] = val_metrics["loss"]

    # Run training
    trainer.run(train_loader, max_epochs=config.MAX_EPOCHS)

    # Save outputs
    if config.SAVE_MODEL:
        torch.save(model.state_dict(), config.MODEL_SAVE_PATH)

    if config.SAVE_RESULTS:
        np.savetxt(
            config.RESULTS_PATH,
            log_matrix,
            fmt='%.4f',
            delimiter=',',
            header=','.join([
                'train_acc', 'train_loss',
                *[f'val{i + 1}_acc,val{i + 1}_loss' for i in range(len(val_loaders))]
            ])
        )

    return log_matrix


def main():
    """Centralized training workflow"""
    model = initialize_model(config.MODEL_TYPE)
    train_loader, val_loaders = create_data_loaders()
    return train_pipeline(model)


if __name__ == '__main__':
    # Initialize hydraulic graph structure
    main()
