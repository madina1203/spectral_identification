import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import lightning.pytorch as pl
from torch.utils.data import DataLoader, Subset
import numpy as np
from lightning.pytorch.loggers import CSVLogger
from sklearn.model_selection import train_test_split

# Import your custom dataset and model
from src.transformers.lance_dataset import CustomSpectrumDataset  # Update to your module path
from src.transformers.model_lance import SimpleSpectraTransformer

# Set the seed for reproducibility
SEED = 1
pl.seed_everything(SEED, workers=True)

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train a model using a Lance dataset with internal batching')
    parser.add_argument('--lance_path', type=str, required=True, help='Path to the Lance dataset')
    parser.add_argument('--epochs', type=int, default=2, help='Number of epochs to train')
    # Recommend using at most 1 worker per DepthCharge best practice
    parser.add_argument('--num_workers', type=int, default=1, help='Number of worker processes (should be <= 1)')
    parser.add_argument('--log_dir', type=str, default='csv_logs', help='Directory to save logs')
    parser.add_argument('--d_model', type=int, default=64, help='Dimension of the model')
    parser.add_argument('--n_layers', type=int, default=2, help='Number of transformer layers')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    # Internal batching performed by the dataset (e.g., how many spectra are processed per record batch)
    parser.add_argument('--internal_batch_size', type=int, default=1024,
                        help='Batch size used by the dataset internally')
    args = parser.parse_args()

    # Setup CSV logger
    csv_logger = CSVLogger(args.log_dir, name="spectra_transformer_experiment_lance_07_05")

    import time

    # Create the dataset instance using the custom from_lance method.
    # Note: The dataset's batch_size is used internally to batch and pad spectra.
    full_dataset = CustomSpectrumDataset.from_lance(
        path=args.lance_path,
        batch_size=1024
    )

    print(f"Total samples in full dataset: {len(full_dataset)}")
    # Optionally, check a sample for required keys.
    sample = full_dataset[0]
    for key in ['mz_array', 'intensity_array', 'instrument_settings', 'label', 'precursor_mz']:
        if key not in sample:
            print(f"Missing key: {key}")

    # Split dataset indices for training and validation
    indices = np.arange(len(full_dataset))
    train_indices, val_indices = train_test_split(indices, test_size=0.3, random_state=SEED)
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    print(f"Number of samples in training dataset: {len(train_dataset)}")
    print(f"Number of samples in validation dataset: {len(val_dataset)}")

    # Create DataLoaders.
    # Since the dataset itself handles batching and padding, set batch_size=1 in the DataLoader.
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,  # Each item is a pre-batched record from the dataset.
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Profiling: measure time to load the first batch from train_loader
    t0 = time.time()
    first_batch = next(iter(train_loader))
    t1 = time.time()
    print(f"[PROFILE] Time to load first batch from train_loader: {t1 - t0:.4f} seconds")

    # Initialize your model
    model = SimpleSpectraTransformer(
        d_model=args.d_model,
        n_layers=args.n_layers,
        dropout=args.dropout,
        lr=args.lr
    )

    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("No GPU available, using CPU")

    total_params = sum(p.numel() for p in model.parameters())
    print("Total parameters:", total_params)

    # Setup PyTorch Lightning Trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        logger=csv_logger,
        log_every_n_steps=10,
        accelerator='gpu',  # Automatically use GPU if available
        devices='auto'
    )

    # Profiling: measure total training time
    train_start = time.time()
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    train_end = time.time()
    print(f"[PROFILE] Total training time (trainer.fit): {train_end - train_start:.2f} seconds")

    # Save the trained model
    model_save_path = os.path.join(args.log_dir, "model.pt")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
