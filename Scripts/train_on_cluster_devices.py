import os

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch  # Import PyTorch
import torch.nn as nn  # Import PyTorch's neural network module
import torch.nn.functional as F  # Import PyTorch's functional module
import lightning.pytorch as pl  # Import PyTorch Lightning
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchsummary import summary
import numpy as np
import argparse
from src.transformers.simpleDataset import SimpleMassSpecDataset
from src.transformers.model_simplified import SimpleSpectraTransformer
from torch.utils.data import ConcatDataset
from lightning.pytorch.loggers import CSVLogger
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from torch.profiler import profile, record_function, ProfilerActivity
from lightning.pytorch.strategies import DDPStrategy

import time

# Set the seed for reproducibility
SEED =1
pl.seed_everything(SEED, workers=True)

# Define a custom collate function to handle variable-length sequences
def collate_fn(batch):
    mz_arrays = [torch.tensor(item['mz_array'], dtype=torch.float32) for item in batch]
    intensity_arrays = [torch.tensor(item['intensity_array'], dtype=torch.float32) for item in batch]
    # Convert list of instrument settings arrays to a single NumPy array first
    instrument_settings = np.array([item['instrument_settings'] for item in batch], dtype=np.float32)
    # # Then convert the NumPy array to a PyTorch tensor
    instrument_settings = torch.tensor(instrument_settings, dtype=torch.float32)

    #print("Instrument settings tensor:", instrument_settings)


    labels = torch.tensor([item['label'] for item in batch], dtype=torch.float32).view(-1, 1)
    precursor_mz = torch.tensor([item['precursor_mz'] for item in batch], dtype=torch.float32).view(-1, 1)
    # Pad sequences to the maximum length in the batch
    max_len = max(mz.shape[0] for mz in mz_arrays)
    mz_padded = torch.zeros(len(batch), max_len)
    intensity_padded = torch.zeros(len(batch), max_len)

    for i in range(len(batch)):
        length = mz_arrays[i].shape[0]
        mz_padded[i, :length] = mz_arrays[i]
        intensity_padded[i, :length] = intensity_arrays[i]

    return {
        'mz': mz_padded,
        'intensity': intensity_padded,
        'instrument_settings': instrument_settings,
        'labels': labels,
        'precursor_mz': precursor_mz
    }

def measure_dataloader(loader, num_batches=None):
    """
    Iterate over the DataLoader and time how long it takes to load batches.
    If num_batches is provided, only load that many batches.
    """
    t0 = time.time()
    for i, batch in enumerate(loader):
        if num_batches is not None and i + 1 >= num_batches:
            break
    total_time = time.time() - t0
    num_loaded = i + 1
    print(f"Loaded {num_loaded} batches in {total_time:.2f} seconds, "
          f"average time per batch: {total_time / num_loaded:.2f} seconds.")


def main(args):
    csv_logger = CSVLogger(args.log_dir, name="spectra_transformer_experiment_22_03")

    # mzml_files = args.mzml_files.split(",")
    # csv_files = args.csv_files.split(",")
    # Read file paths from file_paths.txt
    with open(args.file_paths, 'r') as f:
        lines = f.readlines()
    mzml_files = [line.split(",")[0].strip() for line in lines]
    csv_files = [line.split(",")[1].strip() for line in lines]
    datasets = [SimpleMassSpecDataset(mzml, csv) for mzml, csv in zip(mzml_files, csv_files)]
    combined_dataset = ConcatDataset(datasets)
    labels = [combined_dataset[i]['label'] for i in range(len(combined_dataset))]
    labels = np.array(labels, dtype=np.int64)

    train_indices, val_indices = train_test_split(
        np.arange(len(combined_dataset)), test_size=args.val_split, random_state=args.seed
    )

    train_dataset = Subset(combined_dataset, train_indices)
    val_dataset = Subset(combined_dataset, val_indices)

    batch_size = args.batch_size
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        persistent_workers=True,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        persistent_workers=True,
        pin_memory=True
    )

    # Measure data loading time (e.g., for 10 batches)
    measure_dataloader(train_loader, num_batches=10)
    model = SimpleSpectraTransformer(
        d_model=args.d_model,
        n_layers=args.n_layers,
        dropout=args.dropout,
        lr=args.lr,
        instrument_embedding_dim=args.instrument_embedding_dim
    )
    # model = model.to("cuda")
    # print("Model is on device:", next(model.parameters()).device)
    # summary(model)
    # Add callbacks for saving the best model and early stopping
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename="best-checkpoint",
        verbose=True
    )

    early_stopping_callback = pl.callbacks.EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=10,
        verbose=True
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=2,
        strategy=DDPStrategy(gradient_as_bucket_view=True, static_graph=True),
        precision='16-mixed',
        max_epochs=args.epochs,
        logger=csv_logger,
        log_every_n_steps=args.log_steps,
        callbacks=[checkpoint_callback, early_stopping_callback]
    )
    # Прямой вызов fit без профилировщика
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_paths", type=str, required=True, help="Path to file containing mzML and CSV paths")
    parser.add_argument("--log_dir", type=str, default="./logs", help="Directory for logs")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size")
    parser.add_argument("--val_split", type=float, default=0.3, help="Validation split")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of DataLoader workers")
    parser.add_argument("--d_model", type=int, default=128, help="Model dimension")
    parser.add_argument("--n_layers", type=int, default=2, help="Number of transformer layers")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--log_steps", type=int, default=10, help="Log every n steps")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--instrument_embedding_dim", type=int, default=16, help="Dimension of the instrument embedding output")
    args = parser.parse_args()
    main(args)
