import csv
import time
import os


def setup_log_file():
    os.makedirs("logs", exist_ok=True)
    log_file = f"logs/training_{time.strftime('%Y-%m-%d-%H-%M')}.csv"
    with open(log_file, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss"])
    return log_file


def log_metrics(log_file, epoch, train_loss, val_loss):
    with open(log_file, "a") as f:
        writer = csv.writer(f)
        writer.writerow([epoch, train_loss, val_loss])
