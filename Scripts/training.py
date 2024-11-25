import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


from src.transformers.model import SpectraTransformerWithInstrumentSettings
import torch  # Import PyTorch
import torch.nn as nn  # Import PyTorch's neural network module
import torch.nn.functional as F  # Import PyTorch's functional module
import lightning.pytorch as pl  # Import PyTorch Lightning
from torch.utils.data import DataLoader, random_split, Subset
import numpy as np
from src.transformers.CustomDataset import MassSpecDataset
from torch.utils.data import ConcatDataset
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.loggers import CSVLogger
from sklearn.model_selection import StratifiedShuffleSplit

# Set the seed for reproducibility
SEED = 42  #
pl.seed_everything(SEED, workers=True)

def collate_fn(batch):
    mz_arrays = [torch.tensor(item['mz_array'], dtype=torch.float32) for item in batch]
    intensity_arrays = [torch.tensor(item['intensity_array'], dtype=torch.float32) for item in batch]
    # Convert list of instrument settings arrays to a single NumPy array first
    instrument_settings = np.array([item['instrument_settings'] for item in batch], dtype=np.float32)
    # Then convert the NumPy array to a PyTorch tensor
    instrument_settings = torch.tensor(instrument_settings, dtype=torch.float32)


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
        'instrument_settings': instrument_settings,
        'labels': labels
    }








if __name__ == "__main__":
    # Instantiate the dataset
    # dataset = MassSpecDataset(
    #     mzml_file='/Users/madinabekbergenova/Desktop/phd_data/methods_by_orbitraps/Exploris/MSV_Files/MSV000095364/raw/preserv_etoh_blank_ID_03.mzML',
    #     csv_file='/Users/madinabekbergenova/Desktop/phd_data/methods_by_orbitraps/Exploris/MSV_Files/MSV000095364/raw/preserv_etoh_blank_ID_03_processed_annotated.csv'
    # )
    # dataloader = DataLoader(combined_dataset, batch_size=32, shuffle=True)
    csv_logger = CSVLogger("csv_logs", name="spectra_transformer_experiment")

    #trying to create datasets from several mzml and csv files
    # Lists of mzML and CSV files
    mzml_files = ['/Users/madinabekbergenova/Desktop/phd_data/methods_by_orbitraps/Exploris/MSV_Files/MSV000095364/raw/preserv_etoh_blank_ID_05.mzML',  '/Users/madinabekbergenova/Desktop/phd_data/methods_by_orbitraps/Exploris/MSV_Files/MSV000095364/raw/QCmix.mzML']
    csv_files = ['/Users/madinabekbergenova/Desktop/phd_data/methods_by_orbitraps/Exploris/MSV_Files/MSV000095364/raw/preserv_etoh_blank_ID_05_processed_annotated.csv', '/Users/madinabekbergenova/Desktop/phd_data/methods_by_orbitraps/Exploris/MSV_Files/MSV000095364/raw/QCmix_processed_annotated.csv']

    datasets = [MassSpecDataset(mzml, csv) for mzml, csv in zip(mzml_files, csv_files)]
    combined_dataset = ConcatDataset(datasets)
    # Calculate the sizes for training and validation datasets
    # dataset_size = len(combined_dataset)
    # val_size = int(0.2 * dataset_size)  # For a 20% validation split
    # train_size = dataset_size - val_size
    # Extract labels for stratification
    labels = [combined_dataset[i]['label'] for i in range(len(combined_dataset))]

    # Stratified split
    stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_indices, val_indices = next(stratified_split.split(np.zeros(len(labels)), labels))
    # Split the combined dataset
    # train_dataset, val_dataset = random_split(combined_dataset, [train_size, val_size])
    #splitting the dataset for training and validation
    # Create the DataLoader with the custom collate function
    #dataloader = DataLoader(combined_dataset, batch_size=8, shuffle=True, num_workers=7, collate_fn=collate_fn, persistent_workers=True)
    # Create training and validation subsets
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
    # model = SpectraTransformerWithInstrumentSettings(d_model=64, n_layers=4, dropout=0.1)
     # Define the trainer
    #simplified model
    model = SpectraTransformerWithInstrumentSettings(d_model=64, n_layers = 2, dropout=0.1)
    trainer = pl.Trainer(max_epochs=5, logger=csv_logger)
    #  Train the model
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
# Example batch data (batch size = 16, n_peaks = 100, d_model = 128)
# batch = {
#     "mz": torch.randn(16, 100),  # Example m/z input
#     "intensity": torch.randn(16, 100),  # Example intensity input
#     "instrument_settings": torch.randn(16, 27),  # Example instrument settings input
#     "labels": torch.randint(0, 2, (16,))  # Binary labels (0/1) for each spectrum
# }

# Forward pass
# output = model(batch)
# print(output)
