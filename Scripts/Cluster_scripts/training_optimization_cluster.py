import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
print("Current sys.path:")
print("\n".join(sys.path))

import torch
import lightning.pytorch as pl
from torch.utils.data import DataLoader, Subset, ConcatDataset
import numpy as np
from sklearn.model_selection import train_test_split
from smac import Scenario
from ConfigSpace import ConfigurationSpace, Categorical, UniformFloatHyperparameter, UniformIntegerHyperparameter
from smac import MultiFidelityFacade as MFFacade
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from src.transformers.simpleDataset import SimpleMassSpecDataset
from src.transformers.model_simplified import SimpleSpectraTransformer
from argparse import ArgumentParser

SEED = 1
pl.seed_everything(SEED, workers=True)

# Define a custom collate function to handle variable-length sequences
def collate_fn(batch):
    mz_arrays = [torch.tensor(item['mz_array'], dtype=torch.float32) for item in batch]
    intensity_arrays = [torch.tensor(item['intensity_array'], dtype=torch.float32) for item in batch]
    instrument_settings = np.array([item['instrument_settings'] for item in batch], dtype=np.float32)
    instrument_settings = torch.tensor(instrument_settings, dtype=torch.float32)

    labels = torch.tensor([item['label'] for item in batch], dtype=torch.float32).view(-1, 1)
    precursor_mz = torch.tensor([item['precursor_mz'] for item in batch], dtype=torch.float32).view(-1, 1)
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

# Prepare the DataLoader
def prepare_dataloaders(file_paths, batch_size, val_split, num_workers):
    with open(file_paths, 'r') as f:
        lines = f.readlines()
    mzml_files = [line.split(",")[0].strip() for line in lines]
    csv_files = [line.split(",")[1].strip() for line in lines]
    datasets = [SimpleMassSpecDataset(mzml, csv) for mzml, csv in zip(mzml_files, csv_files)]
    combined_dataset = ConcatDataset(datasets)

    train_indices, val_indices = train_test_split(
        np.arange(len(combined_dataset)), test_size=val_split, random_state=SEED
    )
    train_dataset = Subset(combined_dataset, train_indices)
    val_dataset = Subset(combined_dataset, val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    return train_loader, val_loader

# Define the SMAC Target Function
def train_model_smac(config, seed=None, budget=None):
    d_model = config["d_model"]
    lr = config["learning_rate"]
    optimizer_name = config["optimizer_name"]
    n_layers = config["n_layers"]
    instrument_embedding_dim = config["instrument_embedding_dim"]
    hidden_fc1 =config["hidden_fc1"]
    dropout = config["dropout"]
    if seed is not None:
        pl.seed_everything(seed, workers=True)
    batch_size = args.batch_size

    train_loader, val_loader = prepare_dataloaders(
        file_paths=args.file_paths,
        batch_size=batch_size,
        val_split=args.val_split,
        num_workers=args.num_workers
    )

    model = SimpleSpectraTransformer(
        d_model=d_model,
        n_layers=n_layers,
        dropout=dropout,
        lr=lr,
        hidden_fc1=hidden_fc1,
        instrument_embedding_dim=instrument_embedding_dim,
        encoder_lr=lr,
        linear_lr=lr,
        weight_decay=0.001,
        optimizer_name=optimizer_name

    )

    early_stopping = EarlyStopping(monitor="val_loss", patience=10, mode="min", verbose=True)
    model_checkpoint = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1, verbose=True)

    csv_logger = CSVLogger(save_dir="logs", name="smac_optimization_12_05")
    max_epochs = int(budget) if budget else 15

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        logger=csv_logger,
        callbacks=[early_stopping, model_checkpoint]
    )

    trainer.fit(model, train_loader, val_loader)
    val_metrics = trainer.validate(model, val_loader, verbose=False)
    val_loss = val_metrics[0]["val_loss"]
    return val_loss

# Run SMAC Optimization
def run_smac_optimization():
    cs = ConfigurationSpace()
    cs.add_hyperparameter(Categorical("d_model", [64, 128, 256]))
    cs.add_hyperparameter(UniformFloatHyperparameter("learning_rate", 1e-4, 1e-2, log=True))
    cs.add_hyperparameter(Categorical("optimizer_name", ["Adam", "AdamW"]))
    cs.add_hyperparameter(Categorical("hidden_fc1", [8, 16, 32, 64]))
    cs.add_hyperparameter(UniformIntegerHyperparameter("n_layers", 2, 6))
    cs.add_hyperparameter(Categorical("instrument_embedding_dim", [8, 16, 32]))
    cs.add_hyperparameter(UniformFloatHyperparameter("dropout", 0.1, 0.5, log=True))

    scenario = Scenario(
        configspace=cs,
        name="optimization_cluster_12_05",
        n_trials=50,
        objectives=["val_loss"],
        deterministic=True,
        min_budget=8,
        max_budget=50
    )

    smac = MFFacade(
        scenario=scenario,
        target_function=train_model_smac,
        overwrite=False
    )
    incumbent = smac.optimize()
    print("SMAC optimization finished successfully.")
    print("Best Configuration:", incumbent)

    # best_val_loss = train_model_smac(incumbent)
    # print("Best Validation Loss:", best_val_loss)

    # incumbent_cost = smac.validate(incumbent)
    # print(f"Incumbent cost: {incumbent_cost}")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--file_paths", type=str, required=True, help="Path to file containing mzML and CSV paths")
    parser.add_argument("--log_dir", type=str, default="./logs", help="Directory for logs")
    parser.add_argument("--val_split", type=float, default=0.3, help="Validation split")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of DataLoader workers")
    parser.add_argument("--log_steps", type=int, default=10, help="Log every n steps")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    args = parser.parse_args()

    run_smac_optimization()
