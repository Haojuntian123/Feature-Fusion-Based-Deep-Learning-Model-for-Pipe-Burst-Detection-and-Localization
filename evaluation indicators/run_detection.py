# coding=utf-8
# @name:        run_detection.py
# @software:    PyCharm
# @description: Deep learning-based pipeline anomaly detection system

import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from model.location.RNN_model import MyRNN
from model.location.CNN_model import MyCNN
from model.location.FADenseNet import FADenseNet
from datas.dataset import MyAlarmData
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, ConfusionMatrix, ClassificationReport
import config  # Import configuration settings

# Load seed setting from config
torch.manual_seed(config.SEED)
torch.cuda.manual_seed(config.SEED)
device = config.DEVICE


def run(
        model,
        optimizer,
        loss_fn,
        train_loader,
        test_loader,
        max_epochs,
        is_save=False,
        fname=config.MODEL_SAVE_PATH):
    """
    Supervised training pipeline with performance monitoring.
    """
    trainer = create_supervised_trainer(model, optimizer, loss_fn, device=device)
    val_metrics = {
        "accuracy": Accuracy(device=device),
        "loss": Loss(loss_fn, device=device),
        "cr": ClassificationReport(device=device),
        "cm": ConfusionMatrix(2, device=device)
    }
    evaluator = create_supervised_evaluator(model, metrics=val_metrics, device=device)

    log_matrix = torch.zeros((max_epochs, 12))

    @trainer.on(Events.EPOCH_COMPLETED)
    def process_epoch(trainer):
        evaluator.run(train_loader)
        train_metrics = evaluator.state.metrics
        train_report = eval(train_metrics["cr"])

        evaluator.run(test_loader)
        eval_metrics = evaluator.state.metrics
        eval_report = eval(eval_metrics["cr"])

        # Record metrics
        epoch_idx = trainer.state.epoch - 1
        log_matrix[epoch_idx, 0] = train_metrics["accuracy"]
        log_matrix[epoch_idx, 1] = train_metrics["loss"]
        log_matrix[epoch_idx, 2] = train_report['1']['precision']
        log_matrix[epoch_idx, 3] = train_report['1']['recall']
        log_matrix[epoch_idx, 4] = train_report['0']['precision']
        log_matrix[epoch_idx, 5] = train_report['0']['recall']
        log_matrix[epoch_idx, 6] = eval_metrics["accuracy"]
        log_matrix[epoch_idx, 7] = eval_metrics["loss"]
        log_matrix[epoch_idx, 8] = eval_report['1']['precision']
        log_matrix[epoch_idx, 9] = eval_report['1']['recall']
        log_matrix[epoch_idx, 10] = eval_report['0']['precision']
        log_matrix[epoch_idx, 11] = eval_report['0']['recall']

    trainer.run(train_loader, max_epochs=max_epochs)

    if is_save:
        torch.save(model.state_dict(), fname)

    if config.SAVE_RESULTS:
        np.savetxt(
            config.RESULT_FILE_PATH,
            log_matrix.numpy(),
            fmt='%.4f',
            delimiter=',',
            header='train_acc,train_loss,train_abn_prec,train_abn_rec,train_norm_prec,train_norm_rec,eval_acc,eval_loss,eval_abn_prec,eval_abn_rec,eval_norm_prec,eval_norm_rec'
        )

    return log_matrix


def initialize_model(model_type):
    """Model factory based on configuration"""
    if model_type == 'RNN':
        return MyRNN(
            config.IN_FEATURES,
            config.GRU_HIDDEN,
            config.LINEAR_HIDDEN,
            config.OUT_FEATURES
        ).to(device)
    elif model_type == 'CNN':
        return MyCNN(
            config.IN_CHANNELS,
            config.HIDDEN_CHANNEL1,
            config.HIDDEN_CHANNEL2,
            config.OUT_CHANNEL,
            config.HIDDEN_FEATURES,
            config.OUT_FEATURES
        ).to(device)
    elif model_type == 'FADenseNet':
        return FADenseNet(
            config.IN_FEATURES,
            config.OUT_FEATURES
        ).to(device)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def prepare_dataloader(model_type):
    """Prepare data loaders based on config"""
    norm_file = config.TRAIN_NORM_FILE if model_type == 'train' else config.TEST_NORM_FILE
    burst_file = config.TRAIN_BURST_FILE if model_type == 'train' else config.TEST_BURST_FILE

    dataset = MyAlarmData(
        norm_file,
        burst_file,
        need_time_len=config.SERIES_LENGTH,
        left_offset=config.LEFT_OFFSET,
        right_offset=config.RIGHT_OFFSET
    )

    if config.DATA_SPLIT:
        train_size = int(config.TRAIN_RATIO * len(dataset))
        train_data, eval_data = random_split(dataset, [train_size, len(dataset) - train_size])
        return (
            DataLoader(train_data, config.BATCH_SIZE, shuffle=True),
            DataLoader(eval_data, config.BATCH_SIZE, shuffle=True)
        )
    return DataLoader(dataset, config.BATCH_SIZE, shuffle=True), None


def train_model():
    """Main training workflow."""
    # Initialize model
    model = initialize_model(config.MODEL_TYPE)

    # Prepare data
    train_loader, test_loader = prepare_dataloader('train')
    if test_loader is None:  # Handle non-split case
        _, test_loader = prepare_dataloader('test')

    # Configure optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    class_weights = torch.tensor(config.CLASS_WEIGHTS, device=device, dtype=torch.float32)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    # Execute training
    return run(
        model,
        optimizer,
        loss_fn,
        train_loader,
        test_loader,
        config.MAX_EPOCHS,
        config.SAVE_MODEL,
        fname=config.MODEL_SAVE_PATH
    )


def predict(model, input_data):
    """Run inference using trained model"""
    model.eval()
    with torch.no_grad():
        return model(input_data)


if __name__ == '__main__':
    train_model()
