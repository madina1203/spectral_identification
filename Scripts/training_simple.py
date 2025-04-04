import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
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
from torch.profiler import profile, record_function, ProfilerActivity

# Set the seed for reproducibility
SEED =1
pl.seed_everything(SEED, workers=True)

# Define a custom collate function to handle variable-length sequences
# def collate_fn(batch):
#     mz_arrays = [torch.tensor(item['mz_array'], dtype=torch.float32) for item in batch]
#     intensity_arrays = [torch.tensor(item['intensity_array'], dtype=torch.float32) for item in batch]
#     # Convert list of instrument settings arrays to a single NumPy array first
#     instrument_settings = np.array([item['instrument_settings'] for item in batch], dtype=np.float32)
#     # # Then convert the NumPy array to a PyTorch tensor
#     instrument_settings = torch.tensor(instrument_settings, dtype=torch.float32)
#
#
#     labels = torch.tensor([item['label'] for item in batch], dtype=torch.float32).view(-1, 1)
#     precursor_mz = torch.tensor([item['precursor_mz'] for item in batch], dtype=torch.float32).view(-1, 1)
#     # Pad sequences to the maximum length in the batch
#     max_len = max(mz.shape[0] for mz in mz_arrays)
#     mz_padded = torch.zeros(len(batch), max_len)
#     intensity_padded = torch.zeros(len(batch), max_len)
#
#     for i in range(len(batch)):
#         length = mz_arrays[i].shape[0]
#         mz_padded[i, :length] = mz_arrays[i]
#         intensity_padded[i, :length] = intensity_arrays[i]
#
#     return {
#         'mz': mz_padded,
#         'intensity': intensity_padded,
#         'instrument_settings': instrument_settings,
#         'labels': labels,
#         'precursor_mz': precursor_mz
#     }
#
def collate_fn(batch):
    import numpy as np
    import torch

    # Convert lists of NumPy arrays to tensors using from_numpy (faster, no copy unless needed)
    mz_arrays = [torch.from_numpy(item['mz_array']).float() for item in batch]
    intensity_arrays = [torch.from_numpy(item['intensity_array']).float() for item in batch]

    # Instrument settings: already a list of fixed-length arrays → convert to single NumPy, then to tensor
    instrument_settings = np.array([item['instrument_settings'] for item in batch], dtype=np.float32)
    instrument_settings = torch.from_numpy(instrument_settings)

    # Labels and precursor_mz: simple scalar values per sample
    labels = torch.from_numpy(np.array([item['label'] for item in batch], dtype=np.float32)).view(-1, 1)
    precursor_mz = torch.from_numpy(np.array([item['precursor_mz'] for item in batch], dtype=np.float32)).view(-1, 1)

    # Padding: find max length
    max_len = max(mz.shape[0] for mz in mz_arrays)
    batch_size = len(batch)

    # Pre-allocate padded tensors
    mz_padded = torch.zeros(batch_size, max_len, dtype=torch.float32)
    intensity_padded = torch.zeros(batch_size, max_len, dtype=torch.float32)

    for i in range(batch_size):
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
        '/Users/madinabekbergenova/Desktop/phd_data/methods_by_orbitraps/Exploris/MSV_Files/MSV000087935/POS_MSMS_raw/DOM_Interlab-LCMS_Lab26_A_Pos_MS2_rep2.mzML',
        '/Users/madinabekbergenova/Desktop/phd_data/methods_by_orbitraps/Exploris/MSV_Files/MSV000095364/raw/QCmix.mzML',
        '/Users/madinabekbergenova/Desktop/phd_data/methods_by_orbitraps/Exploris/MSV_Files/MSV000088226/Fraction-6-4.mzML'

    ]


    csv_files = [
        '/Users/madinabekbergenova/Desktop/phd_data/methods_by_orbitraps/Exploris/MSV_Files/MSV000087935/POS_MSMS_raw/DOM_Interlab-LCMS_Lab26_A_Pos_MS2_rep2_processed_annotated.csv',
        '/Users/madinabekbergenova/Desktop/phd_data/methods_by_orbitraps/Exploris/MSV_Files/MSV000095364/raw/QCmix_processed_annotated.csv',
        '/Users/madinabekbergenova/Desktop/phd_data/methods_by_orbitraps/Exploris/MSV_Files/MSV000088226/Fraction-6-4_processed_annotated.csv'
    ]

    datasets = [SimpleMassSpecDataset(mzml, csv) for mzml, csv in zip(mzml_files, csv_files)]
    combined_dataset = ConcatDataset(datasets)
    labels = [combined_dataset[i]['label'] for i in range(len(combined_dataset))]
    labels = np.array(labels, dtype=np.int64)

    for i, sample in enumerate(combined_dataset):
        # print(f"Sample {i}: {sample}")
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
    print(f"Number of samples in the full dataset: {len(combined_dataset)}")
    # print(f"Number of training samples: {len(train_indices)}")
    # print(f"Number of validation samples: {len(val_indices)}")

    model = SimpleSpectraTransformer(
        d_model=64,
        n_layers=2,   # Adjust based on your needs
        dropout=0.3,
        lr=0.0001
    )
    #moving the model to mps device
    #device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device("cpu")
    print(f"Using device: {device}")

    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print("Total parameters:", total_params)

    trainer = pl.Trainer(max_epochs=5, logger=csv_logger, log_every_n_steps=10  # Log every 10 steps, or set to 1 for every batch
)
    # Wrap the training process with PyTorch's profiler
    with profile(
            activities=[ProfilerActivity.CPU],
            profile_memory=True,
            record_shapes=True
    ) as prof:
        with record_function("model_training"):
            trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Print a profiling summary table
    print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))
    # Optionally, you can also print CPU time usage:
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))