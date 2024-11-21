import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch  # Import PyTorch
import torch.nn as nn  # Import PyTorch's neural network module
import torch.nn.functional as F  # Import PyTorch's functional module
import lightning.pytorch as pl  # Import PyTorch Lightning
from torch.utils.data import DataLoader, Subset
import numpy as np
from src.transformers.simpleDataset import SimpleMassSpecDataset
from src.transformers.model_simplified import SimpleSpectraTransformer
from torch.utils.data import ConcatDataset
from lightning.pytorch.loggers import CSVLogger
from sklearn.model_selection import StratifiedShuffleSplit

# Set the seed for reproducibility
SEED = 42
pl.seed_everything(SEED, workers=True)

# Define a custom collate function to handle variable-length sequences
def collate_fn(batch):
    mz_arrays = [torch.tensor(item['mz_array'], dtype=torch.float32) for item in batch]
    intensity_arrays = [torch.tensor(item['intensity_array'], dtype=torch.float32) for item in batch]
    # Convert list of instrument settings arrays to a single NumPy array first
    # instrument_settings = np.array([item['instrument_settings'] for item in batch], dtype=np.float32)
    # # Then convert the NumPy array to a PyTorch tensor
    # instrument_settings = torch.tensor(instrument_settings, dtype=torch.float32)


    labels = torch.tensor([item['label'] for item in batch], dtype=torch.float32).view(-1, 1)

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
        # 'instrument_settings': instrument_settings,
        'labels': labels
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
    print(len(combined_dataset))
    for i, sample in enumerate(combined_dataset):
        print(f"Sample {i}: {sample}")
        if 'mz_array' not in sample or 'intensity_array' not in sample or 'label' not in sample:
            print(f"Missing keys in sample {i}")
    stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_indices, val_indices = next(stratified_split.split(np.zeros(len(labels)), labels))

    train_dataset = Subset(combined_dataset, train_indices)
    val_dataset = Subset(combined_dataset, val_indices)
    batch_size = 8

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
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

    model = SimpleSpectraTransformer(
        d_model=128,  # Adjust based on your needs
        n_layers=4,   # Adjust based on your needs
        dropout=0.1,
        lr=0.001
    )

    trainer = pl.Trainer(max_epochs=10, logger=csv_logger)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
