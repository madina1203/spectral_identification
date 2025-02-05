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
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
# SMAC imports (v2.2.0)
# ---------------------------
from smac import  Scenario
from ConfigSpace import ConfigurationSpace, Categorical
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, CategoricalHyperparameter, UniformIntegerHyperparameter
from smac import MultiFidelityFacade as MFFacade
from smac import RunHistory
from matplotlib import pyplot as plt
# ---------------------------
# My custom modules
# ---------------------------
from src.transformers.simpleDataset import SimpleMassSpecDataset
from src.transformers.model_simplified import SimpleSpectraTransformer
from lightning.pytorch.loggers import CSVLogger
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

def prepare_dataloaders():
    mzml_files = [
        '/Users/madinabekbergenova/Desktop/phd_data/methods_by_orbitraps/Exploris/MSV_Files/MSV000095364/raw/QCmix.mzML'
    ]

    csv_files = [
         '/Users/madinabekbergenova/Desktop/phd_data/methods_by_orbitraps/Exploris/MSV_Files/MSV000095364/raw/QCmix_processed_annotated.csv'
    ]

    # Create datasets and combine
    datasets = [SimpleMassSpecDataset(mzml, csv) for mzml, csv in zip(mzml_files, csv_files)]
    combined_dataset = ConcatDataset(datasets)

    # Split indices for train/val
    all_indices = np.arange(len(combined_dataset))
    train_indices, val_indices = train_test_split(all_indices, test_size=0.3, random_state=SEED)

    train_dataset = Subset(combined_dataset, train_indices)
    val_dataset = Subset(combined_dataset, val_indices)

    # Example: set a batch size
    batch_size = 64

    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # or use your WeightedRandomSampler here if needed
        num_workers=4,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )

    return train_loader, val_loader

# ---------------------------
# 3. Define SMAC Target Function
# ---------------------------
def train_model_smac(config,seed=None, budget=None):
    """
    SMAC will call this function with a dictionary 'config' that has:
      - config['learning_rate']
      - config['d_model']

    We train for only 5 epochs to quickly verify SMAC works with your setup.
    Return the validation loss (lower is better).
    """
    #d_model = config["d_model"]
    # hidden_fc1_choice = config["hidden_fc1_choice"]
    # if hidden_fc1_choice == "d_model":
    #     hidden_fc1 = d_model
    # else:  # "2xd_model"
    #     hidden_fc1 = 2 * d_model
    # d_model = config["d_model"]
    lr = config["learning_rate"]
    n_layers = config["n_layers"]
    # optimizer_name = config["optimizer_name"]
    if seed is not None:
        pl.seed_everything(seed, workers=True)
    # Prepare data
    train_loader, val_loader = prepare_dataloaders()

    # Define your model with these hyperparameters
    model = SimpleSpectraTransformer(
        d_model=64,
        n_layers=n_layers,
        dropout=0.3,
        lr=lr,
        hidden_fc1=64,
        encoder_lr=lr,
        linear_lr=lr,
        weight_decay=0.001,
        optimizer_name="Adam",
    )

    # Define early stopping and model checkpoint callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',  # Metric to monitor
        patience=4,  # Stop if no improvement for 2 consecutive epochs
        mode='min',  # Minimize val_loss
        verbose=True
    )

    model_checkpoint = ModelCheckpoint(
        monitor='val_loss',  # Save based on validation loss
        mode='min',  # Minimize val_loss
        save_top_k=1,  # Keep only the best model
        filename='best_model',  # Name for the best model checkpoint
        verbose=True
    )

    # Add a CSV logger to log metrics
    logger = CSVLogger(
        save_dir='logs',  # Directory where logs will be saved
        name='smac_model_training_04.02_2',  # Subdirectory for this training run

    )
    max_epochs = int(budget) if budget else 15
    # Define a quick trainer (5 epochs)
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        logger=logger,
        enable_checkpointing=True,  # Enable checkpointing
        callbacks=[early_stopping, model_checkpoint]
    )

    # Train
    trainer.fit(model, train_loader, val_loader)

    # Load the best model for evaluation
    best_model_path = model_checkpoint.best_model_path


    # Validate to get val_loss
    val_metrics = trainer.validate(model, val_loader, verbose=False)
    # val_metrics is typically a list of dicts (one for each dataloader)
    # e.g., [{'val_loss': 0.1234, ...}]
    val_loss = val_metrics[0]['val_loss']

    # SMAC tries to *minimize* this objective
    return val_loss


# ---------------------------
# 4. Run SMAC Optimization
# ---------------------------
def run_smac_optimization():
    # A) Define your hyperparameter space
    cs = ConfigurationSpace()
    # Add hyperparameters to the configuration space
    #learning_rate = UniformFloatHyperparameter("learning_rate", 0.0001, 0.001, default_value=0.001)
    cs.add_hyperparameter(UniformFloatHyperparameter("learning_rate", 1e-4, 1e-3, log=True))
    #cs.add_hyperparameter(Categorical("d_model", [64, 128]))
    cs.add_hyperparameter(UniformIntegerHyperparameter("n_layers", 2,6))
    # B) Create a scenario
    #
    scenario = Scenario(
        configspace=cs,
        n_trials=3,
        objectives=["val_loss"],
        deterministic=True,  # set False if your data/ training is stochastic
        min_budget = 2,  # minimum number of epochs
        max_budget = 5  # maximum number of epochs
    )

    # C) Set up SMAC
    smac = MFFacade(
        scenario=scenario,
        target_function=train_model_smac
    )

    # D) Start the optimization
    incumbent = smac.optimize()  # Best config found

    print("SMAC optimization finished successfully.")
   
    # Re-run or directly compute the val_loss with the best config
    print("Best Configuration :", incumbent)


   




       

if __name__ == '__main__':
    run_smac_optimization()
