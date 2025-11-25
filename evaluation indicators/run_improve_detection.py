# coding=utf-8
# @name:        run_improve_detection.py
# @software:    PyCharm
# @description: Multi-sensor fusion pipeline for anomaly detection

import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from model.improve_location.IRNN import IRNN
from model.improve_location.ICNN import ICNN
from model.improve_location.IFA import IFA
from datas.dataset import MyAlarmDataQ
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, ConfusionMatrix, ClassificationReport
import config  # Import configuration module

# Configuration-based initialization
torch.manual_seed(config.SEED)
torch.cuda.manual_seed(config.SEED)
device = config.DEVICE


def train_model(model, train_loader, test_loader):
    """
    Model training pipeline with metrics tracking.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    class_weights = torch.tensor(config.CLASS_WEIGHTS, device=device, dtype=config.DTYPE)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    trainer = create_supervised_trainer(model, optimizer, loss_fn, device=device)
    metrics = {
        "accuracy": Accuracy(device=device),
        "loss": Loss(loss_fn, device=device),
        "cr": ClassificationReport(device=device),
        "cm": ConfusionMatrix(config.N_CLASSES, device=device)
    }
    evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)

    log_matrix = np.zeros((config.MAX_EPOCHS, config.N_METRICS))

    @trainer.on(Events.EPOCH_COMPLETED)
    def record_metrics(engine):
        # Evaluate training set
        evaluator.run(train_loader)
        train_metrics = evaluator.state.metrics
        train_cr = eval(train_metrics["cr"])

        # Evaluate validation set
        evaluator.run(test_loader)
        valid_metrics = evaluator.state.metrics
        valid_cr = eval(valid_metrics["cr"])

        epoch = engine.state.epoch - 1
        # Training metrics
        log_matrix[epoch, 0] = train_metrics["accuracy"]
        log_matrix[epoch, 1] = train_metrics["loss"]
        log_matrix[epoch, 2] = train_cr[1]['precision']
        log_matrix[epoch, 3] = train_cr[1]['recall']
        # Validation metrics
        log_matrix[epoch, 6] = valid_metrics["accuracy"]
        log_matrix[epoch, 7] = valid_metrics["loss"]
        log_matrix[epoch, 8] = valid_cr[1]['precision']
        log_matrix[epoch, 9] = valid_cr[1]['recall']

    trainer.run(train_loader, max_epochs=config.MAX_EPOCHS)

    if config.SAVE_MODEL:
        torch.save(model.state_dict(), config.MODEL_SAVE_PATH)

    if config.SAVE_RESULTS:
        np.savetxt(
            config.RESULTS_PATH,
            log_matrix,
            fmt='%.4f',
            delimiter=',',
            header='train_acc,train_loss,train_abn_prec,train_abn_rec,_,_,val_acc,val_loss,val_abn_prec,val_abn_rec,_,_'
        )
    return log_matrix


def get_dataloaders():
    """Prepare pressure-flow fused dataset"""
    dataset = MyAlarmDataQ(
        norm_p_file=config.TRAIN_NORM_PRESSURE,
        norm_q_file=config.TRAIN_NORM_FLOW,
        burst_p_file=config.TRAIN_BURST_PRESSURE,
        burst_q_file=config.TRAIN_BURST_FLOW,
        need_time_len=config.SEQUENCE_LENGTH,
        left_offset=config.TIME_OFFSETS[0],
        right_offset=config.TIME_OFFSETS[1]
    )
    train_size = int(len(dataset) * config.TRAIN_RATIO)
    train_set, val_set = random_split(
        dataset,
        [train_size, len(dataset) - train_size]
    )
    return (
        DataLoader(train_set, config.BATCH_SIZE, shuffle=True),
        DataLoader(val_set, config.BATCH_SIZE, shuffle=False)
    )


def create_sensor_graph():
    """Generate flow-pressure adjacency matrix"""
    wn = wntr.network.WaterNetworkModel(config.HYDRAULIC_MODEL)
    cg = CreateGraph(wn)
    pressure_nodes = cg.index2id(config.PRESSURE_SENSORS)
    return torch.from_numpy(
        cg.cal_g(config.FLOW_SENSORS, pressure_nodes)
    ).to(device)


def initialize_model(model_type):
    """Model factory based on configuration"""
    if model_type == 'IRNN':
        return IRNN(
            config.GRAPH,
            config.N_FEATURES,
            config.GRU_HIDDEN,
            config.LINEAR_DIM,
            config.FLOW_FUSION_DIM,
            config.N_CLASSES
        ).to(device)
    elif model_type == 'ICNN':
        return ICNN(
            config.GRAPH,
            len(config.PRESSURE_SENSORS),
            config.IN_CHANNELS,
            config.CHANNELS[0],
            config.CHANNELS[1],
            config.OUT_CHANNELS,
            config.LATENT_DIM,
            config.N_CLASSES
        ).to(device)
    elif model_type == 'IFA':
        return IFA(
            config.GRAPH,
            len(config.PRESSURE_SENSORS),
            config.N_FEATURES,
            config.N_CLASSES,
            config.SEQUENCE_LENGTH
        ).to(device)


def main():
    """Centralized training workflow"""
    train_loader, test_loader = get_dataloaders()
    model = initialize_model(config.MODEL_TYPE)
    metrics = train_model(model, train_loader, test_loader)
    return metrics


if __name__ == '__main__':
    # Initialize global graph structure
    config.GRAPH = create_sensor_graph()
    main()
