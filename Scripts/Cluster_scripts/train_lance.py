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
from lightning.pytorch.loggers import CSVLogger
from sklearn.model_selection import train_test_split
from torch.profiler import profile, record_function, ProfilerActivity

# Import your custom dataset and model
from src.transformers.lance_dataset import CustomSpectrumDataset  # update to your actual module path
from src.transformers.model_lance import SimpleSpectraTransformer

# Set the seed for reproducibility
SEED = 1
pl.seed_everything(SEED, workers=True)


def collate_fn(batch):
    import torch
    import torch.nn as nn
    import numpy as np

    # Get the keys from the first sample in the batch.
    keys = batch[0].keys()
    collated = {}
    
    # Special handling for variable-length arrays (mz_array and intensity_array)
    if 'mz_array' in keys:
        # Get all mz_arrays from the batch and ensure they're 1D tensors
        mz_arrays = []
        for sample in batch:
            arr = sample['mz_array']
            # Convert to tensor if not already
            if not isinstance(arr, torch.Tensor):
                arr = torch.tensor(arr, dtype=torch.float32)
            # Ensure it's a 1D tensor
            if arr.dim() > 1:
                arr = arr.squeeze()
            mz_arrays.append(arr)
        
        # Use nn.utils.rnn.pad_sequence for padding (like in depthcharge)
        collated['mz_array'] = nn.utils.rnn.pad_sequence(mz_arrays, batch_first=True)
    
    if 'intensity_array' in keys:
        # Get all intensity_arrays from the batch and ensure they're 1D tensors
        intensity_arrays = []
        for sample in batch:
            arr = sample['intensity_array']
            # Convert to tensor if not already
            if not isinstance(arr, torch.Tensor):
                arr = torch.tensor(arr, dtype=torch.float32)
            # Ensure it's a 1D tensor
            if arr.dim() > 1:
                arr = arr.squeeze()
            intensity_arrays.append(arr)
        
        # Use nn.utils.rnn.pad_sequence for padding (like in depthcharge)
        collated['intensity_array'] = nn.utils.rnn.pad_sequence(intensity_arrays, batch_first=True)
    
    # Handle other keys that can be stacked directly
    for key in keys:
        if key not in ['mz_array', 'intensity_array']:
            try:
                # Try to stack the tensors
                collated[key] = torch.stack([sample[key] for sample in batch])
            except:
                # If stacking fails, convert to tensor if possible
                try:
                    values = [sample[key] for sample in batch]
                    if all(isinstance(v, (int, float)) for v in values):
                        collated[key] = torch.tensor(values, dtype=torch.float32)
                    else:
                        # For more complex types, just keep the list
                        collated[key] = values
                except:
                    # If all else fails, just keep the list
                    collated[key] = [sample[key] for sample in batch]

    # Optionally, if your model expects different key names, you can rename them:
    result = {
        'mz': collated['mz_array'],  # padded m/z arrays
        'intensity': collated['intensity_array'],  # padded intensity arrays
        'instrument_settings': collated['instrument_settings'],
    }
    
    # Add labels if present
    if 'label' in collated:
        if isinstance(collated['label'], torch.Tensor):
            result['labels'] = collated['label'].view(-1, 1)
        else:
            result['labels'] = torch.tensor(collated['label'], dtype=torch.float32).view(-1, 1)
    
    # Add precursor_mz if present
    if 'precursor_mz' in collated:
        if isinstance(collated['precursor_mz'], torch.Tensor):
            result['precursor_mz'] = collated['precursor_mz'].view(-1, 1)
        else:
            result['precursor_mz'] = torch.tensor(collated['precursor_mz'], dtype=torch.float32).view(-1, 1)
    
    return result


if __name__ == '__main__':
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train a model using Lance dataset')
    parser.add_argument('--lance_path', type=str, required=True,
                        help='Path to the Lance dataset')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=2,
                        help='Number of epochs to train')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of worker processes for data loading')
    parser.add_argument('--log_dir', type=str, default='csv_logs',
                        help='Directory to save logs')
    parser.add_argument('--d_model', type=int, default=64,
                        help='Dimension of the model')
    parser.add_argument('--n_layers', type=int, default=2,
                        help='Number of transformer layers')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate')
    
    args = parser.parse_args()
    
    # Setup logger
    csv_logger = CSVLogger(args.log_dir, name="spectra_transformer_experiment")

    # Use the provided Lance file path
    lance_file_path = args.lance_path
    batch_size = args.batch_size

    # Create the dataset instance using your custom dataset's from_lance method.
    full_dataset = CustomSpectrumDataset.from_lance(
        path=lance_file_path,
        batch_size=batch_size
    )

    # Optionally, you can check the length and a few sample keys.
    print(f"Total samples in full dataset: {len(full_dataset)}")
    sample = full_dataset[0]
    for key in ['mz_array', 'intensity_array', 'instrument_settings', 'label', 'precursor_mz']:
        if key not in sample:
            print(f"Missing key: {key}")

    # Split the dataset into training and validation using indices.
    indices = np.arange(len(full_dataset))
    train_indices, val_indices = train_test_split(
        indices, test_size=0.3, random_state=SEED
    )
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    # Calculate class weights for the sampler if needed
    train_labels = [int(full_dataset[i]['label'].item()) for i in train_indices]

    class_counts = np.bincount(train_labels)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[label] for label in train_labels]

    # Create DataLoaders with the custom collate function.
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # No shuffle for validation
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        persistent_workers=True
    )

    print(f"Number of samples in training dataset: {len(train_dataset)}")
    print(f"Number of samples in validation dataset: {len(val_dataset)}")

    # Initialize your model
    model = SimpleSpectraTransformer(
        d_model=args.d_model,
        n_layers=args.n_layers,
        dropout=args.dropout,
        lr=args.lr
    )

    # Check for GPU availability
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("No GPU available, using CPU")

    total_params = sum(p.numel() for p in model.parameters())
    print("Total parameters:", total_params)

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        logger=csv_logger,
        log_every_n_steps=10,  # Log every 10 steps (or adjust as needed)
        accelerator='auto',    # Automatically use GPU if available
        devices='auto'         # Use all available devices
    )

    # Train the model
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    
    # Save the model
    model_save_path = os.path.join(args.log_dir, "model.pt")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
