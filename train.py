import argparse
import datetime
import glob
import os
import shutil
import time
from math import ceil

import numpy as np
import psutil
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader

from dataset import TrainDataset, TrainData, TestData, TestDataset
from metrics import FocalLoss, f1_score, f1_score_from_probs, F1Loss
from models import ResNet, Ensemble, SimpleCnn
from utils import get_learning_rate, str2bool, adjust_learning_rate, adjust_initial_learning_rate, \
    list_sorted_model_files, check_model_improved, log_args, log

cudnn.enabled = True
cudnn.benchmark = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

CLASS_FREQUENCIES = [
    12885.0, 1254.0, 3621.0, 1561.0, 1858.0, 2513.0, 1008.0, 2822.0, 53.0, 45.0, 28.0, 1093.0, 688.0, 537.0, 1066.0,
    21.0, 530.0, 210.0, 902.0, 1482.0, 172.0, 3777.0, 802.0, 2965.0, 322.0, 8228.0, 328.0, 11.0
]

CLASS_WEIGHTS = [
    0.07411719053162592, 0.7615629984051037, 0.26373929853631595, 0.6117873158231902, 0.5139935414424112,
    0.3800238758456029, 0.9474206349206349, 0.33841247342310415, 18.0188679245283, 21.22222222222222,
    34.107142857142854, 0.8737419945105215, 1.3880813953488371, 1.7783985102420856, 0.8958724202626641,
    45.476190476190474, 1.8018867924528301, 4.5476190476190474, 1.058758314855876, 0.6443994601889339,
    5.5523255813953485, 0.25284617421233785, 1.1907730673316708, 0.3220910623946037, 2.9658385093167703,
    0.11606708799222168, 2.9115853658536586, 86.81818181818181
]

CLASS_WEIGHTS_TENSOR = torch.tensor(CLASS_WEIGHTS).float().to(device)


def create_model(type, input_size, num_classes):
    if type == "cnn":
        model = SimpleCnn(num_classes=num_classes)
    elif type in ["resnet18", "resnet34", "resnet50"]:
        model = ResNet(type=type, num_classes=num_classes)
    else:
        raise Exception("Unsupported model type: '{}".format(type))

    return nn.DataParallel(model)


def zero_item_tensor():
    return torch.tensor(0.0).float().to(device, non_blocking=True)


def evaluate(model, data_loader, criterion):
    model.eval()

    loss_sum_t = zero_item_tensor()
    score_sum_t = zero_item_tensor()
    step_count = 0

    with torch.no_grad():
        for batch in data_loader:
            images, categories = \
                batch[0].to(device, non_blocking=True), \
                batch[1].to(device, non_blocking=True)

            prediction_logits = model(images)
            criterion.weight = CLASS_WEIGHTS_TENSOR
            loss = criterion(prediction_logits, categories)

            loss_sum_t += loss
            score_sum_t += f1_score(prediction_logits, categories)

            step_count += 1

    loss_avg = loss_sum_t.item() / step_count
    score_avg = score_sum_t.item() / step_count

    return loss_avg, score_avg


def create_criterion(loss_type):
    if loss_type == "bce":
        criterion = nn.BCEWithLogitsLoss()
    elif loss_type == "focal":
        criterion = FocalLoss(gamma=2)
    elif loss_type == "f1":
        criterion = F1Loss()
    else:
        raise Exception("Unsupported loss type: '{}".format(loss_type))
    return criterion


def create_optimizer(type, model, lr):
    if type == "adam":
        return optim.Adam(model.parameters(), lr=lr)
    elif type == "sgd":
        return optim.SGD(model.parameters(), lr=lr, weight_decay=1e-4, momentum=0.9, nesterov=True)
    else:
        raise Exception("Unsupported optimizer type: '{}".format(type))


def predict(model, data_loader):
    model.eval()

    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for batch in data_loader:
            images = batch[0].to(device, non_blocking=True)
            predictions = torch.sigmoid(model(images)).cpu().data.numpy()
            all_predictions.extend(predictions)
            if len(batch) > 1:
                all_targets.extend(batch[1].cpu().data.numpy())

    return all_predictions, all_targets


def load_ensemble_model(base_dir, ensemble_model_count, data_loader, criterion, model_type, input_size, num_classes):
    ensemble_model_candidates = list_sorted_model_files(base_dir)[-(2 * ensemble_model_count):]
    if os.path.isfile("{}/swa_model.pth".format(base_dir)):
        ensemble_model_candidates.append("{}/swa_model.pth".format(base_dir))

    score_to_model = {}
    for model_file_path in ensemble_model_candidates:
        model_file_name = os.path.basename(model_file_path)
        model = create_model(type=model_type, input_size=input_size, num_classes=num_classes).to(device)
        model.load_state_dict(torch.load(model_file_path, map_location=device))

        val_loss_avg, val_score_avg = evaluate(model, data_loader, criterion)
        log("ensemble '%s': val_loss=%.4f, val_score=%.4f" % (model_file_name, val_loss_avg, val_score_avg))

        if len(score_to_model) < ensemble_model_count or min(score_to_model.keys()) < val_score_avg:
            if len(score_to_model) >= ensemble_model_count:
                del score_to_model[min(score_to_model.keys())]
            score_to_model[val_score_avg] = model

    ensemble = Ensemble(list(score_to_model.values()))

    val_loss_avg, val_score_avg = evaluate(ensemble, data_loader, criterion)
    log("ensemble: val_loss=%.4f, val_score=%.4f" % (val_loss_avg, val_score_avg))

    return ensemble


def calculate_categories_from_predictions(predictions, threshold):
    return [np.squeeze(np.argwhere(p > threshold), axis=1) for p in predictions]


def calculate_best_threshold(predictions, targets):
    predictions_t = torch.tensor(predictions)
    targets_t = torch.tensor(targets)

    thresholds = np.linspace(0, 1, 51)
    scores = [f1_score_from_probs(predictions_t, targets_t, threshold=t) for t in thresholds]

    best_score_index = np.argmax(scores)

    return thresholds[best_score_index], scores[best_score_index]


def main():
    args = argparser.parse_args()
    log_args(args)

    input_dir = args.input_dir
    output_dir = args.output_dir
    base_model_dir = args.base_model_dir
    image_size = args.image_size
    augment = args.augment
    use_progressive_image_sizes = args.use_progressive_image_sizes
    progressive_image_size_min = args.progressive_image_size_min
    progressive_image_size_step = args.progressive_image_size_step
    progressive_image_epoch_step = args.progressive_image_epoch_step
    batch_size = args.batch_size
    batch_iterations = args.batch_iterations
    num_workers = args.num_workers
    pin_memory = args.pin_memory
    epochs_to_train = args.epochs
    lr_scheduler_type = args.lr_scheduler
    lr_patience = args.lr_patience
    lr_min = args.lr_min
    lr_max = args.lr_max
    lr_min_decay = args.lr_min_decay
    lr_max_decay = args.lr_max_decay
    optimizer_type = args.optimizer
    loss_type = args.loss
    model_type = args.model
    patience = args.patience
    sgdr_cycle_epochs = args.sgdr_cycle_epochs
    sgdr_cycle_epochs_mult = args.sgdr_cycle_epochs_mult
    sgdr_cycle_end_prolongation = args.sgdr_cycle_end_prolongation
    sgdr_cycle_end_patience = args.sgdr_cycle_end_patience
    max_sgdr_cycles = args.max_sgdr_cycles

    progressive_image_sizes = list(range(progressive_image_size_min, image_size + 1, progressive_image_size_step))

    train_data = TrainData(input_dir)

    train_set = TrainDataset(train_data.train_set_df, input_dir, 28, image_size)
    train_set_data_loader = \
        DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                   pin_memory=pin_memory)

    val_set = TrainDataset(train_data.val_set_df, input_dir, 28, image_size)
    val_set_data_loader = \
        DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    if base_model_dir:
        for base_file_path in glob.glob("{}/*.pth".format(base_model_dir)):
            shutil.copyfile(base_file_path, "{}/{}".format(output_dir, os.path.basename(base_file_path)))
        model = create_model(type=model_type, input_size=image_size, num_classes=28).to(device)
        model.load_state_dict(torch.load("{}/model.pth".format(output_dir), map_location=device))
        optimizer = create_optimizer(optimizer_type, model, lr_max)
        if os.path.isfile("{}/optimizer.pth".format(output_dir)):
            optimizer.load_state_dict(torch.load("{}/optimizer.pth".format(output_dir)))
            adjust_initial_learning_rate(optimizer, lr_max)
            adjust_learning_rate(optimizer, lr_max)
    else:
        model = create_model(type=model_type, input_size=image_size, num_classes=28).to(device)
        optimizer = create_optimizer(optimizer_type, model, lr_max)

    torch.save(model.state_dict(), "{}/model.pth".format(output_dir))

    ensemble_model_index = 0
    for model_file_path in glob.glob("{}/model-*.pth".format(output_dir)):
        model_file_name = os.path.basename(model_file_path)
        model_index = int(model_file_name.replace("model-", "").replace(".pth", ""))
        ensemble_model_index = max(ensemble_model_index, model_index + 1)

    epoch_iterations = ceil(len(train_set) / batch_size)

    log("train_set_samples: {}, val_set_samples: {}".format(len(train_set), len(val_set)))
    log()

    global_val_score_best_avg = float("-inf")
    sgdr_cycle_val_score_best_avg = float("-inf")

    lr_scheduler = CosineAnnealingLR(optimizer, T_max=sgdr_cycle_epochs, eta_min=lr_min)

    optim_summary_writer = SummaryWriter(log_dir="{}/logs/optim".format(output_dir))
    train_summary_writer = SummaryWriter(log_dir="{}/logs/train".format(output_dir))
    val_summary_writer = SummaryWriter(log_dir="{}/logs/val".format(output_dir))

    current_sgdr_cycle_epochs = sgdr_cycle_epochs
    sgdr_next_cycle_end_epoch = current_sgdr_cycle_epochs + sgdr_cycle_end_prolongation
    sgdr_iterations = 0
    sgdr_cycle_count = 0
    batch_count = 0
    epoch_of_last_improval = 0

    lr_scheduler_plateau = \
        ReduceLROnPlateau(optimizer, mode="max", min_lr=lr_min, patience=lr_patience, factor=0.8, threshold=1e-4)

    log('{"chart": "best_val_score", "axis": "epoch"}')
    log('{"chart": "val_score", "axis": "epoch"}')
    log('{"chart": "val_loss", "axis": "epoch"}')
    log('{"chart": "sgdr_cycle", "axis": "epoch"}')
    log('{"chart": "score", "axis": "epoch"}')
    log('{"chart": "loss", "axis": "epoch"}')
    log('{"chart": "lr_scaled", "axis": "epoch"}')
    log('{"chart": "mem_used", "axis": "epoch"}')
    log('{"chart": "epoch_time", "axis": "epoch"}')

    train_start_time = time.time()

    criterion = create_criterion(loss_type)

    for epoch in range(epochs_to_train):
        epoch_start_time = time.time()

        log("memory used: {:.2f} GB".format(psutil.virtual_memory().used / 2 ** 30))

        if use_progressive_image_sizes:
            next_image_size = \
                progressive_image_sizes[min(epoch // progressive_image_epoch_step, len(progressive_image_sizes) - 1)]

            if train_set.image_size != next_image_size:
                log("changing image size to {}".format(next_image_size))
                train_set.image_size = next_image_size
                val_set.image_size = next_image_size

        model.train()

        train_loss_sum_t = zero_item_tensor()
        train_score_sum_t = zero_item_tensor()

        epoch_batch_iter_count = 0

        if lr_scheduler_type == "lr_finder":
            r = current_sgdr_cycle_epochs / min(current_sgdr_cycle_epochs, (sgdr_iterations + 1) / epoch_iterations)
            new_lr = lr_min + r * (lr_max - lr_min)
            adjust_learning_rate(optimizer, new_lr)

        for b, batch in enumerate(train_set_data_loader):
            images, categories = \
                batch[0].to(device, non_blocking=True), \
                batch[1].to(device, non_blocking=True)

            if lr_scheduler_type == "cosine_annealing":
                lr_scheduler.step(epoch=min(current_sgdr_cycle_epochs, sgdr_iterations / epoch_iterations))

            if b % batch_iterations == 0:
                optimizer.zero_grad()

            prediction_logits = model(images)
            criterion.weight = CLASS_WEIGHTS_TENSOR
            loss = criterion(prediction_logits, categories)
            loss.backward()

            with torch.no_grad():
                train_loss_sum_t += loss
                train_score_sum_t += f1_score(prediction_logits, categories)

            if (b + 1) % batch_iterations == 0 or (b + 1) == len(train_set_data_loader):
                optimizer.step()

            sgdr_iterations += 1
            batch_count += 1
            epoch_batch_iter_count += 1

            optim_summary_writer.add_scalar("lr", get_learning_rate(optimizer), batch_count + 1)

        train_loss_avg = train_loss_sum_t.item() / epoch_batch_iter_count
        train_score_avg = train_score_sum_t.item() / epoch_batch_iter_count

        val_loss_avg, val_score_avg = evaluate(model, val_set_data_loader, criterion)

        if lr_scheduler_type == "reduce_on_plateau":
            lr_scheduler_plateau.step(val_score_avg)

        model_improved_within_sgdr_cycle = check_model_improved(sgdr_cycle_val_score_best_avg, val_score_avg)
        if model_improved_within_sgdr_cycle:
            torch.save(model.state_dict(), "{}/model-{}.pth".format(output_dir, ensemble_model_index))
            sgdr_cycle_val_score_best_avg = val_score_avg

        model_improved = check_model_improved(global_val_score_best_avg, val_score_avg)
        ckpt_saved = False
        if model_improved:
            torch.save(model.state_dict(), "{}/model.pth".format(output_dir))
            torch.save(optimizer.state_dict(), "{}/optimizer.pth".format(output_dir))
            global_val_score_best_avg = val_score_avg
            epoch_of_last_improval = epoch
            ckpt_saved = True

        sgdr_reset = False
        if (lr_scheduler_type == "cosine_annealing") \
                and (epoch + 1 >= sgdr_next_cycle_end_epoch) \
                and (epoch - epoch_of_last_improval >= sgdr_cycle_end_patience):
            sgdr_iterations = 0
            current_sgdr_cycle_epochs = int(current_sgdr_cycle_epochs * sgdr_cycle_epochs_mult)
            sgdr_next_cycle_end_epoch = epoch + 1 + current_sgdr_cycle_epochs + sgdr_cycle_end_prolongation

            ensemble_model_index += 1
            sgdr_cycle_val_score_best_avg = float("-inf")
            sgdr_cycle_count += 1
            sgdr_reset = True

            new_lr_min = lr_min * (lr_min_decay ** sgdr_cycle_count)
            new_lr_max = lr_max * (lr_max_decay ** sgdr_cycle_count)
            new_lr_max = max(new_lr_max, new_lr_min)

            adjust_learning_rate(optimizer, new_lr_max)
            lr_scheduler = CosineAnnealingLR(optimizer, T_max=current_sgdr_cycle_epochs, eta_min=new_lr_min)

        optim_summary_writer.add_scalar("sgdr_cycle", sgdr_cycle_count, epoch + 1)

        train_summary_writer.add_scalar("loss", train_loss_avg, epoch + 1)
        train_summary_writer.add_scalar("score", train_score_avg, epoch + 1)
        val_summary_writer.add_scalar("loss", val_loss_avg, epoch + 1)
        val_summary_writer.add_scalar("score", val_score_avg, epoch + 1)

        epoch_end_time = time.time()
        epoch_duration_time = epoch_end_time - epoch_start_time

        log(
            "[%03d/%03d] %ds, lr: %.6f, loss: %.4f, val_loss: %.4f, score: %.4f, val_score: %.4f, ckpt: %d, rst: %d" % (
                epoch + 1,
                epochs_to_train,
                epoch_duration_time,
                get_learning_rate(optimizer),
                train_loss_avg,
                val_loss_avg,
                train_score_avg,
                val_score_avg,
                int(ckpt_saved),
                int(sgdr_reset)))

        log('{"chart": "best_val_score", "x": %d, "y": %.4f}' % (epoch + 1, global_val_score_best_avg))
        log('{"chart": "val_loss", "x": %d, "y": %.4f}' % (epoch + 1, val_loss_avg))
        log('{"chart": "val_score", "x": %d, "y": %.4f}' % (epoch + 1, val_score_avg))
        log('{"chart": "sgdr_cycle", "x": %d, "y": %d}' % (epoch + 1, sgdr_cycle_count))
        log('{"chart": "loss", "x": %d, "y": %.4f}' % (epoch + 1, train_loss_avg))
        log('{"chart": "score", "x": %d, "y": %.4f}' % (epoch + 1, train_score_avg))
        log('{"chart": "lr_scaled", "x": %d, "y": %.4f}' % (epoch + 1, 1000 * get_learning_rate(optimizer)))
        log('{"chart": "mem_used", "x": %d, "y": %.2f}' % (epoch + 1, psutil.virtual_memory().used / 2 ** 30))
        log('{"chart": "epoch_time", "x": %d, "y": %d}' % (epoch + 1, epoch_duration_time))

        if (sgdr_reset or lr_scheduler_type == "reduce_on_plateau") and epoch - epoch_of_last_improval >= patience:
            log("early abort due to lack of improval")
            break

        if max_sgdr_cycles is not None and sgdr_cycle_count >= max_sgdr_cycles:
            log("early abort due to maximum number of sgdr cycles reached")
            break

    optim_summary_writer.close()
    train_summary_writer.close()
    val_summary_writer.close()

    train_end_time = time.time()
    log()
    log("Train time: %s" % str(datetime.timedelta(seconds=train_end_time - train_start_time)))

    model.load_state_dict(torch.load("{}/model.pth".format(output_dir), map_location=device))

    val_predictions, val_targets = predict(model, val_set_data_loader)
    best_threshold, best_score = calculate_best_threshold(val_predictions, val_targets)
    log("Best threshold / score: {} / {}".format(best_threshold, best_score))

    test_data = TestData(input_dir)
    test_set = TestDataset(test_data.test_set_df, input_dir, image_size)
    test_set_data_loader = \
        DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    submission_df = test_data.test_set_df.copy()
    test_predictions, _ = predict(model, test_set_data_loader)
    predicted_categories = calculate_categories_from_predictions(test_predictions, threshold=best_threshold)
    submission_df["Predicted"] = [" ".join(map(str, pc)) for pc in predicted_categories]
    submission_df.to_csv("{}/submission.csv".format(output_dir))


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--input_dir", default="/storage/kaggle/hpa")
    argparser.add_argument("--output_dir", default="/artifacts")
    argparser.add_argument("--base_model_dir", default=None)
    argparser.add_argument("--image_size", default=128, type=int)
    argparser.add_argument("--augment", default=False, type=str2bool)
    argparser.add_argument("--use_progressive_image_sizes", default=False, type=str2bool)
    argparser.add_argument("--progressive_image_size_min", default=32, type=int)
    argparser.add_argument("--progressive_image_size_step", default=16, type=int)
    argparser.add_argument("--progressive_image_epoch_step", default=7, type=int)
    argparser.add_argument("--epochs", default=500, type=int)
    argparser.add_argument("--batch_size", default=64, type=int)
    argparser.add_argument("--batch_iterations", default=1, type=int)
    argparser.add_argument("--test_size", default=0.1, type=float)
    argparser.add_argument("--fold", default=None, type=int)
    argparser.add_argument("--num_workers", default=8, type=int)
    argparser.add_argument("--pin_memory", default=True, type=str2bool)
    argparser.add_argument("--lr_scheduler", default="cosine_annealing")
    argparser.add_argument("--lr_patience", default=1, type=int)
    argparser.add_argument("--lr_min", default=0.001, type=float)
    argparser.add_argument("--lr_max", default=0.01, type=float)
    argparser.add_argument("--lr_min_decay", default=1.0, type=float)
    argparser.add_argument("--lr_max_decay", default=1.0, type=float)
    argparser.add_argument("--model", default="cnn")
    argparser.add_argument("--patience", default=5, type=int)
    argparser.add_argument("--optimizer", default="sgd")
    argparser.add_argument("--loss", default="f1")
    argparser.add_argument("--sgdr_cycle_epochs", default=5, type=int)
    argparser.add_argument("--sgdr_cycle_epochs_mult", default=1.0, type=float)
    argparser.add_argument("--sgdr_cycle_end_prolongation", default=0, type=int)
    argparser.add_argument("--sgdr_cycle_end_patience", default=0, type=int)
    argparser.add_argument("--max_sgdr_cycles", default=None, type=int)

    main()
