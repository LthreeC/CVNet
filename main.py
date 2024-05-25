import os
import sys
import csv
import numpy as np
from datetime import datetime
import torch
from tqdm import tqdm
import argparse
from dataproc import data_process
from models import choose_model, show_model
from utils import *


class Runner(object):
    def __init__(self, ShowingModel, Device, DatasetPath, ModelName, LossFn, Optimizer, Schedular, Epochs, LrRate, BatchSize,
                 ReSize, ResultDir="results") -> None:
        self.Epochs = Epochs
        self.device = Device
        self.train_dataloader, self.num_classes = data_process(os.path.join(DatasetPath, "train"), ReSize=ReSize, BatchSize=BatchSize)
        self.valid_dataloader, self.num_classes = data_process(os.path.join(DatasetPath, "val"), ReSize=ReSize, BatchSize=BatchSize)
        self.test_dataloader, self.num_classes = data_process(os.path.join(DatasetPath, "test"), ReSize=ReSize, BatchSize=BatchSize)
        # apparently num_classes of train, valid, test are the same

        print("-------------Show data------------")
        print(f"Train: {len(self.train_dataloader.dataset)} samples")
        print(f"Valid: {len(self.valid_dataloader.dataset)} samples")
        print(f"Test: {len(self.test_dataloader.dataset)} samples")
        print(f"Num Classes: {self.num_classes}")
        showing_data(self.train_dataloader)

        print("-------------Starting--------------")
        self.model = choose_model(ModelName, num_classes=self.num_classes).to(self.device)
        if ShowingModel:
            show_model(self.model)
            sys.exit(0)

        self.loss_fn = choose_lossfn(LossFn)
        self.optimizer = choose_optimizer(Optimizer, parameters=self.model.parameters(), lr=LrRate)
        self.scheduler = choose_schedular(Schedular=Schedular, optimizer=self.optimizer, step_size=10, gamma=0.1)

        self.ResultDir = ResultDir
        self.setup_logging()

        # Track best model
        self.best_valid_loss = float('inf')
        self.best_model_path = os.path.join(self.save_mode_dir, "best_model.pth")
        self.final_model_path = os.path.join(self.save_mode_dir, "final_model.pth")

    def setup_logging(self):
        current_time = datetime.now().strftime("%Y_%m%d_%H%M_%S")
        self.run_dir = os.path.join(self.ResultDir, f"ret_{current_time}")
        self.save_mode_dir = os.path.join(self.run_dir, "save_model")
        self.metrics_dir = os.path.join(self.run_dir, "metrics")

        os.makedirs(self.save_mode_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)

        # Initialize CSV files with headers
        with open(os.path.join(self.metrics_dir, "training_log.csv"), 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Mode', 'Epoch', 'Loss',  'Learning Rate', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'Specificity'])
        with open(os.path.join(self.metrics_dir, "validation_log.csv"), 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Mode', 'Epoch', 'Loss',  'Learning Rate', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'Specificity'])
        with open(os.path.join(self.metrics_dir, "testing_log.csv"), 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Mode', 'Epoch', 'Loss',  'Learning Rate', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'Specificity'])

        # Print class parameters to CSV file
        parameters = self.__dict__
        with open(os.path.join(self.run_dir, "class_parameters.csv"), 'w', newline='') as file:
            writer = csv.writer(file)
            for param, value in parameters.items():
                writer.writerow([param, value])

    def write_metrics(self, phase, epoch, loss, metrics, lr, mode):
        accuracy, precision, recall, f1, specificity = metrics
        log_message = [mode, epoch, loss, lr, accuracy * 100, precision * 100, recall * 100, f1 * 100, specificity * 100]
        print(f'{phase} Mode{mode} Epoch {epoch} | Loss: {loss:.4f} | Learning Rate: {lr:.6f} | '
              f'Accuracy: {accuracy * 100:.2f}% | '
              f'Precision: {precision * 100:.2f}% | '
              f'Recall: {recall * 100:.2f}% | '
              f'F1 Score: {f1 * 100:.2f}% | '
              f'Specificity: {specificity * 100:.2f}% | ')

        with open(os.path.join(self.metrics_dir, f"{phase.lower()}_log.csv"), 'a', newline='') as log_file:
            writer = csv.writer(log_file)
            writer.writerow(log_message)

    def train_epoch(self, epoch):
        self.model.train()
        train_loss = 0
        all_targets = []
        all_predictions = []

        for batch_idx, (b_x, b_y) in tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader),
                                          ncols=100):
            b_x, b_y = b_x.to(self.device), b_y.to(self.device)

            self.optimizer.zero_grad()
            b_y_pred = self.model(b_x)
            loss = self.loss_fn(b_y_pred, b_y)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = b_y_pred.max(1)
            all_targets.extend(b_y.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

        train_loss /= len(self.train_dataloader)
        train_metrics = calculate_metrics(all_targets, all_predictions)
        current_lr = self.optimizer.param_groups[0]['lr']
        self.write_metrics('Training', epoch, train_loss, train_metrics, current_lr, 'train')

        return train_loss, train_metrics

    def valid_epoch(self, epoch):
        self.model.eval()
        valid_loss = 0
        all_targets = []
        all_predictions = []

        with torch.no_grad():
            for batch_idx, (b_x, b_y) in tqdm(enumerate(self.valid_dataloader), total=len(self.valid_dataloader),
                                              ncols=100):
                b_x, b_y = b_x.to(self.device), b_y.to(self.device)

                b_y_pred = self.model(b_x)
                loss = self.loss_fn(b_y_pred, b_y)

                valid_loss += loss.item()
                _, predicted = b_y_pred.max(1)
                all_targets.extend(b_y.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

        valid_loss /= len(self.valid_dataloader)
        valid_metrics = calculate_metrics(all_targets, all_predictions)
        current_lr = self.optimizer.param_groups[0]['lr']
        self.write_metrics('Validation', epoch, valid_loss, valid_metrics, current_lr, 'val')

        # Track the best model
        if valid_loss < self.best_valid_loss:
            self.best_valid_loss = valid_loss
            self.best_model_state_dict = self.model.state_dict()  # Save the state dict

        return valid_loss, valid_metrics

    def test_epoch(self, model_path, phase):
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        test_loss = 0
        all_targets = []
        all_predictions = []

        with torch.no_grad():
            for batch_idx, (b_x, b_y) in tqdm(enumerate(self.test_dataloader), total=len(self.test_dataloader), ncols=100):
                b_x, b_y = b_x.to(self.device), b_y.to(self.device)

                b_y_pred = self.model(b_x)
                loss = self.loss_fn(b_y_pred, b_y)

                test_loss += loss.item()
                _, predicted = b_y_pred.max(1)
                all_targets.extend(b_y.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

        test_loss /= len(self.test_dataloader)
        test_metrics = calculate_metrics(all_targets, all_predictions)
        current_lr = self.optimizer.param_groups[0]['lr']
        self.write_metrics('Testing', self.Epochs, test_loss, test_metrics, current_lr, phase)

        # Save targets and predictions for further analysis
        targets_path = os.path.join(self.metrics_dir, f"{phase}_targets.npy")
        predictions_path = os.path.join(self.metrics_dir, f"{phase}_predictions.npy")
        np.save(targets_path, all_targets)
        np.save(predictions_path, all_predictions)

        return test_loss, test_metrics

    def run(self):
        for epoch in range(1, self.Epochs + 1):
            print(f'\nEpoch: {epoch}/{self.Epochs}')
            self.train_epoch(epoch)
            self.valid_epoch(epoch)
            self.scheduler.step()

        # Save final model
        torch.save(self.model.state_dict(), self.final_model_path)
        # Save best model
        torch.save(self.best_model_state_dict, self.best_model_path)
        # Test best model
        print("\nTesting best model:")
        self.test_epoch(self.best_model_path, 'Test_Best')
        # Test final model
        print("\nTesting final model:")
        self.test_epoch(self.final_model_path, 'Test_Final')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Configure your model.')
    parser.add_argument('--ShowingModel', type=int, default=0)
    parser.add_argument('--ModelName', type=str, default='mix')
    parser.add_argument('--DatasetPath', type=str, default=r'datasets/ts_data')
    # parser.add_argument('--ClassNames', nargs='+', default=['AKIEC', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'VASC'])
    parser.add_argument('--Epochs', type=int, default=30)
    parser.add_argument('--ReSize', type=int, default=256)
    parser.add_argument('--BatchSize', type=int, default=32)
    parser.add_argument('--LossFn', type=str, default='cross_entropy')
    parser.add_argument('--Optimizer', type=str, default='adam')
    parser.add_argument('--Scheduler', type=str, default='StepLR')
    parser.add_argument('--LrRate', type=float, default=0.0002)

    args = parser.parse_args()
    Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    runner = Runner(args.ShowingModel, Device, args.DatasetPath, args.ModelName, args.LossFn, args.Optimizer, args.Scheduler,
                    args.Epochs, args.LrRate, args.BatchSize, args.ReSize)
    runner.run()

    # TODO add cross-validation: dataproc, main