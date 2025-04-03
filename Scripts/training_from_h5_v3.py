import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from torch.utils.data import DataLoader, Subset
import numpy as np
import argparse
import h5py
from torch.nn.utils.rnn import pad_sequence
import time
import logging
import psutil
import gc


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("h5_training_memory_load.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("h5_training")

# Set the seed for reproducibility
SEED = 1
pl.seed_everything(SEED, workers=True)



def log_memory_usage():
    process = psutil.Process(os.getpid())
    ram_usage = process.memory_info().rss / (1024 * 1024)  # In MB

    gpu_memory_allocated = 0
    gpu_memory_reserved = 0
    if torch.cuda.is_available():
        gpu_memory_allocated = torch.cuda.memory_allocated() / (1024 * 1024)  # In MB
        gpu_memory_reserved = torch.cuda.memory_reserved() / (1024 * 1024)  # In MB

    logger.info(
        f"Memory usage - RAM: {ram_usage:.2f} MB, GPU allocated: {gpu_memory_allocated:.2f} MB, GPU reserved: {gpu_memory_reserved:.2f} MB")


# Define a custom collate function to handle variable-length sequences
def collate_fn(batch):
    batch_size = len(batch)
    max_mz_length = max(len(item['mz_array']) for item in batch)

    logger.debug(f"Collating batch of size {batch_size}, max sequence length: {max_mz_length}")

    # Creating tensors
    mz_padded = torch.zeros((batch_size, max_mz_length), dtype=torch.float32)
    intensity_padded = torch.zeros((batch_size, max_mz_length), dtype=torch.float32)

    # filling the tensors
    for i, item in enumerate(batch):
        mz_array = torch.tensor(item['mz_array'], dtype=torch.float32)
        intensity_array = torch.tensor(item['intensity_array'], dtype=torch.float32)
        length = len(mz_array)
        mz_padded[i, :length] = mz_array
        intensity_padded[i, :length] = intensity_array

    # Process other data
    instrument_settings = torch.tensor(np.array([item['instrument_settings'] for item in batch]), dtype=torch.float32)
    labels = torch.tensor([item['label'] for item in batch], dtype=torch.float32).view(-1, 1)
    precursor_mz = torch.tensor([item['precursor_mz'] for item in batch], dtype=torch.float32).view(-1, 1)

    # Log tensor shapes periodically (every 100th batch)
    if np.random.random() < 0.01:  # ~1% of batches
        logger.info(f"Batch tensors - mz: {mz_padded.shape}, intensity: {intensity_padded.shape}, "
                    f"instrument_settings: {instrument_settings.shape}, labels: {labels.shape}, "
                    f"precursor_mz: {precursor_mz.shape}")
        # Log label distribution
        positive_labels = (labels > 0.5).sum().item()
        logger.info(f"Label distribution - positive: {positive_labels}/{batch_size} "
                    f"({positive_labels / batch_size * 100:.2f}%)")

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
    logger.info(f"Measuring dataloader performance for {num_batches if num_batches else 'all'} batches...")
    t0 = time.time()

    # Log memory before loading
    log_memory_usage()

    for i, batch in enumerate(loader):
        if i % 10 == 0:  # Log every 10 batches
            elapsed = time.time() - t0
            logger.info(f"Loaded {i + 1} batches in {elapsed:.2f} seconds, "
                        f"average time per batch: {elapsed / (i + 1):.4f} seconds")
            log_memory_usage()

        if num_batches is not None and i + 1 >= num_batches:
            break

    total_time = time.time() - t0
    num_loaded = i + 1
    logger.info(f"TOTAL: Loaded {num_loaded} batches in {total_time:.2f} seconds, "
                f"average time per batch: {total_time / num_loaded:.4f} seconds")

    # Log memory after loading
    log_memory_usage()


# Define a new Dataset class that loads all data into memory
class MemoryH5Dataset(torch.utils.data.Dataset):
    def __init__(self, h5_file_path):
        self.h5_file_path = h5_file_path
        
        logger.info(f"Initializing MemoryH5Dataset with file: {h5_file_path}")
        logger.info("This implementation loads ALL data into memory for maximum performance")
        
        # Get the keys and load all data into memory
        t0 = time.time()
        try:
            # Log initial memory usage
            logger.info("Initial memory usage before loading data:")
            log_memory_usage()
            
            with h5py.File(self.h5_file_path, 'r') as f:
                # Log file structure
                logger.info(f"HDF5 file structure: {list(f.keys())}")

                if 'data_pairs' not in f:
                    logger.error(f"'data_pairs' group not found in HDF5 file. Available groups: {list(f.keys())}")
                    raise KeyError("'data_pairs' group not found in HDF5 file")

                data_group = f['data_pairs']
                self.keys = list(data_group.keys())

                # Log some metadata about the first few items
                if len(self.keys) > 0:
                    sample_key = self.keys[0]
                    sample_group = data_group[sample_key]
                    logger.info(f"Sample item structure: {list(sample_group.keys())}")

                    # Log shapes of arrays in the first item
                    shapes = {k: sample_group[k].shape for k in sample_group.keys()}
                    logger.info(f"Sample item shapes: {shapes}")

                logger.info(f"Found {len(self.keys)} data pairs in HDF5 file")
                
                # Pre-allocate list for all items
                self.data = []
                
                # Load all items into memory with progress reporting
                logger.info("Loading all data into memory...")
                for i, key in enumerate(self.keys):
                    group = data_group[key]
                    
                    item = {
                        'mz_array': group['mz_array'][()],
                        'intensity_array': group['intensity_array'][()],
                        'instrument_settings': group['instrument_settings'][()],
                        'precursor_mz': group['precursor_mz'][()],
                        'label': group['label'][()]
                    }
                    
                    self.data.append(item)
                    
                    # Print progress every 100,000 items
                    if (i + 1) % 100000 == 0:
                        elapsed = time.time() - t0
                        logger.info(f"Loaded {i + 1}/{len(self.keys)} items ({(i + 1)/len(self.keys)*100:.2f}%) "
                                   f"in {elapsed:.2f} seconds...")
                        log_memory_usage()
                
                # Check for label distribution
                if len(self.data) > 0:
                    # Sample a subset of items to check label distribution
                    sample_size = min(1000, len(self.data))
                    sample_indices = np.random.choice(len(self.data), sample_size, replace=False)

                    positive_count = 0
                    for idx in sample_indices:
                        label = self.data[idx]['label']
                        if label > 0.5:  # Assuming binary classification
                            positive_count += 1

                    logger.info(f"Label distribution in sample: {positive_count}/{sample_size} "
                                f"positive ({positive_count / sample_size * 100:.2f}%)")

        except Exception as e:
            logger.error(f"Error initializing MemoryH5Dataset: {str(e)}")
            raise

        load_time = time.time() - t0
        logger.info(f"MemoryH5Dataset initialization completed in {load_time:.2f} seconds")
        logger.info("Final memory usage after loading all data:")
        log_memory_usage()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# Import your model definition
from src.transformers.model_simplified import SimpleSpectraTransformer


def main(args):
    logger.info("=" * 50)
    logger.info(f"Starting training with HDF5 file: {args.h5_file}")
    logger.info(f"Arguments: {args}")
    logger.info("=" * 50)

    # Set up logger
    from lightning.pytorch.loggers import CSVLogger
    csv_logger = CSVLogger(args.log_dir, name="spectra_transformer_experiment_memory_load")

    # Check GPU availability
    logger.info("Checking GPU availability...")
    if torch.cuda.is_available():
        logger.info(f"Available GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            logger.info(f"GPU {i} memory: {torch.cuda.get_device_properties(i).total_memory / (1024 ** 3):.2f} GB")
    else:
        logger.warning("No GPUs available, training will use CPU only")

    # Create the dataset from the .h5 file
    logger.info(f"Loading dataset from HDF5 file: {args.h5_file}")
    t0 = time.time()
    try:
        # Use the memory-based dataset implementation
        dataset = MemoryH5Dataset(args.h5_file)
        logger.info(f"Dataset loaded successfully in {time.time() - t0:.2f} seconds")
        logger.info(f"Dataset size: {len(dataset)} items")
    except Exception as e:
        logger.error(f"Failed to load dataset: {str(e)}")
        raise

    # Split dataset
    logger.info("Splitting dataset into train and validation sets...")
    indices = np.arange(len(dataset))
    train_indices, val_indices = train_test_split(
        indices, test_size=args.val_split, random_state=args.seed
    )

    logger.info(f"Train set size: {len(train_indices)}, Validation set size: {len(val_indices)}")

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    # Create data loaders
    logger.info(f"Creating data loaders with batch_size={args.batch_size}, num_workers={args.num_workers}")
    batch_size = args.batch_size

    # Create batch sampler for more efficient loading
    train_sampler = torch.utils.data.BatchSampler(
        torch.utils.data.RandomSampler(train_dataset),
        batch_size=batch_size,
        drop_last=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,  # Use batch sampler instead of batch_size
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        persistent_workers=True,
        pin_memory=args.pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        persistent_workers=True,
        pin_memory=args.pin_memory
    )

    # Measure data loading performance
    logger.info("Measuring data loading performance...")
    measure_dataloader(train_loader, num_batches=args.measure_batches)

    # Create model
    logger.info(f"Creating model with d_model={args.d_model}, n_layers={args.n_layers}")
    model = SimpleSpectraTransformer(
        d_model=args.d_model,
        n_layers=args.n_layers,
        dropout=args.dropout,
        lr=args.lr,
        instrument_embedding_dim=args.instrument_embedding_dim
    )

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model created with {total_params:,} parameters")

    # Add callbacks
    logger.info("Setting up training callbacks...")
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

    # Create trainer
    logger.info("Creating trainer...")
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        max_epochs=args.epochs,
        logger=csv_logger,
        log_every_n_steps=args.log_steps,
        callbacks=[checkpoint_callback, early_stopping_callback]
    )

    # Start training
    logger.info("Starting model training...")
    t0 = time.time()
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    logger.info(f"Training completed in {(time.time() - t0) / 60:.2f} minutes")
    logger.info(f"Best validation loss: {checkpoint_callback.best_model_score:.4f}")
    logger.info(f"Best model saved at: {checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    from sklearn.model_selection import train_test_split

    parser = argparse.ArgumentParser()
    # Basic parameters
    parser.add_argument("--h5_file", type=str, required=True, help="Path to the .h5 file containing the dataset")
    parser.add_argument("--log_dir", type=str, default="./logs", help="Directory for logs")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size")
    parser.add_argument("--val_split", type=float, default=0.3, help="Validation split")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of DataLoader workers")

    # Model parameters
    parser.add_argument("--d_model", type=int, default=64, help="Model dimension")
    parser.add_argument("--n_layers", type=int, default=2, help="Number of transformer layers")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--instrument_embedding_dim", type=int, default=16,
                        help="Dimension of the instrument embedding output")

    # Training parameters
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--log_steps", type=int, default=10, help="Log every n steps")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")

    # Optimization parameters
    parser.add_argument("--pin_memory", action="store_true", help="Use pinned memory for faster GPU transfer")
    parser.add_argument("--mixed_precision", action="store_true", help="Use mixed precision training")
    parser.add_argument("--devices", type=int, default=1, help="Number of GPU devices to use")
    parser.add_argument("--accumulate_grad_batches", type=int, default=1,
                        help="Number of batches to accumulate gradients")
    parser.add_argument("--measure_batches", type=int, default=10,
                        help="Number of batches to measure for dataloader performance")

    args = parser.parse_args()
    main(args)