import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch  # Import PyTorch
import torch.nn as nn  # Import PyTorch's neural network module
import torch.nn.functional as F  # Import PyTorch's functional module
import lightning.pytorch as pl  # Import PyTorch Lightning
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
import numpy as np
from src.transformers.simpleDataset import SimpleMassSpecDataset
from src.transformers.model_simplified import SimpleSpectraTransformer
from torch.utils.data import ConcatDataset
from lightning.pytorch.loggers import CSVLogger
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
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


if __name__ == '__main__':
    csv_logger = CSVLogger("csv_logs", name="spectra_transformer_experiment")
    mzml_files = [
        '/Users/madinabekbergenova/Desktop/phd_data/methods_by_orbitraps/Exploris/MSV_Files/MSV000095364/raw/preserv_etoh_blank_ID_05.mzML',
        '/Users/madinabekbergenova/Desktop/phd_data/methods_by_orbitraps/Exploris/MSV_Files/MSV000095364/raw/QCmix.mzML'
    ]
    csv_files = [
        '/Users/madinabekbergenova/Desktop/phd_data/methods_by_orbitraps/Exploris/MSV_Files/MSV000095364/raw/preserv_etoh_blank_ID_05_processed_annotated.csv',
        '/Users/madinabekbergenova/Desktop/phd_data/methods_by_orbitraps/Exploris/MSV_Files/MSV000095364/raw/QCmix_processed_annotated.csv'
    ]

    datasets = [SimpleMassSpecDataset(mzml, csv) for mzml, csv in zip(mzml_files, csv_files)]
    combined_dataset = ConcatDataset(datasets)
    labels = [combined_dataset[i]['label'] for i in range(len(combined_dataset))]
    labels = np.array(labels, dtype=np.int64)

    for i, sample in enumerate(combined_dataset):
        print(f"Sample {i}: {sample}")
        if 'mz_array' not in sample or 'intensity_array' not in sample or 'label' not in sample:
            print(f"Missing keys in sample {i}")
    # stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=SEED)
    # train_indices, val_indices = next(stratified_split.split(np.zeros(len(labels)), labels))
    train_indices, val_indices = train_test_split(
        np.arange(len(combined_dataset)), test_size=0.3, random_state=SEED
    )

    train_dataset = Subset(combined_dataset, train_indices)
    val_dataset = Subset(combined_dataset, val_indices)
    # Calculate class weights for the sampler
    train_labels = [combined_dataset[i]['label'] for i in train_indices]
    class_counts = np.bincount(train_labels)  # Count occurrences of each class
    class_weights = 1.0 / class_counts  # Inverse of class frequency
    sample_weights = [class_weights[label] for label in train_labels]  # Weight for each sample in the train dataset

    # Create the WeightedRandomSampler for the training set
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),  # Number of samples to draw
        replacement=True  # Allow replacement to balance classes
    )

    batch_size = 64

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        # sampler=sampler,
        num_workers=7,
        collate_fn=collate_fn,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # No need to shuffle validation data
        num_workers=7,
        collate_fn=collate_fn,
        persistent_workers=True
    )
    # print(f"Number of samples in the full dataset: {len(combined_dataset)}")
    # print(f"Number of training samples: {len(train_indices)}")
    # print(f"Number of validation samples: {len(val_indices)}")

    model = SimpleSpectraTransformer(
        d_model=64,
        n_layers=2,   # Adjust based on your needs
        dropout=0.3,
        lr=0.0001
    )

    trainer = pl.Trainer(max_epochs=50, logger=csv_logger, log_every_n_steps=10  # Log every 10 steps, or set to 1 for every batch
)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
