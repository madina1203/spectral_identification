import argparse
import os
import re

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import spectrum_utils.spectrum as sus
import torch
import torch.nn as nn
from depthcharge.encoders import FloatEncoder
from depthcharge.transformers import SpectrumTransformerEncoder
from pyteomics import mzml
from torch.utils.data import DataLoader, Dataset, Subset, ConcatDataset
from torchmetrics.classification import (BinaryAccuracy, BinaryF1Score,
                                         BinaryPrecision, BinaryRecall)
from sklearn.model_selection import train_test_split


# Model Definition
class MyModel(SpectrumTransformerEncoder):
    """Our custom model class."""

    def __init__(self, *args, **kwargs):
        """Add parameters for the global token hook."""
        super().__init__(*args, **kwargs)
        self.precursor_mz_encoder = FloatEncoder(self.d_model)
        self.apply(self.init_weights)

    def global_token_hook(self, mz_array, intensity_array, precursor_mz=None, *args, **kwargs):
        """Return a simple representation of the precursor."""
        if precursor_mz is None:
            raise ValueError("precursor_mz must be provided in the batch.")
        precursor_mz = precursor_mz.type_as(mz_array).view(-1, 1)
        mz_rep = self.precursor_mz_encoder(precursor_mz)
        if mz_rep.dim() == 3 and mz_rep.shape[1] == 1:
            mz_rep = mz_rep.squeeze(1)
        return mz_rep

    def init_weights(self, module):
        """Custom weight initialization logic."""
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=1)


class SimpleSpectraTransformer(pl.LightningModule):
    """Simplified model using only SpectrumTransformerEncoder for MS1 spectra."""

    def __init__(
            self,
            d_model,
            n_layers,
            dropout=0.1,
            lr=0.001,
            hidden_fc1: int = 64,
            instrument_embedding_dim: int = 16,
            encoder_lr: float = 1e-4,
            linear_lr: float = 1e-3,
            weight_decay: float = 1e-3,
            optimizer_name: str = "AdamW",
    ):
        """Initialize the model with transformer for spectra only."""
        super().__init__()
        lr = float(lr)
        encoder_lr = float(encoder_lr)
        linear_lr = float(linear_lr)
        weight_decay = float(weight_decay)
        self.save_hyperparameters()
        self.d_model = d_model
        self.n_layers = n_layers
        self.lr = lr
        self.hidden_fc1 = hidden_fc1
        self.instrument_embedding_dim = instrument_embedding_dim
        self.encoder_lr = encoder_lr
        self.linear_lr = linear_lr
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer_name
        self.spectrum_encoder = MyModel(
            d_model=d_model,
            n_layers=n_layers,
            dropout=dropout,
        )
        self.fc_instrument_1 = nn.Linear(20, hidden_fc1)
        self.fc_instrument_2 = nn.Linear(hidden_fc1, instrument_embedding_dim)
        self.fc_combined = nn.Linear(d_model + instrument_embedding_dim, d_model)
        self.fc_output = nn.Linear(d_model, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.train_accuracy = BinaryAccuracy()
        self.train_precision = BinaryPrecision()
        self.val_accuracy = BinaryAccuracy()
        self.val_precision = BinaryPrecision()
        self.train_f1 = BinaryF1Score()
        self.train_recall = BinaryRecall()
        self.val_f1 = BinaryF1Score()
        self.val_recall = BinaryRecall()
        self.apply(self.init_weights)
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
        else:
            rank = 0
        print(f"[rank {rank}] model spec: d_model={d_model}  hidden_fc1={hidden_fc1}  "
              f"n_layers={n_layers}  max_seq_len={getattr(self, 'max_len', 'N/A')}",
              flush=True)

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, batch):
        mz_array = batch["mz"].float()
        intensity_array = batch["intensity"].float()
        precursor_mz = batch["precursor_mz"].float()
        spectra_emb, _ = self.spectrum_encoder(mz_array, intensity_array, precursor_mz=precursor_mz)
        spectra_emb = spectra_emb[:, 0, :]
        instrument_settings = batch["instrument_settings"].float()
        instrument_emb = self.fc_instrument_1(instrument_settings)
        instrument_emb = self.relu(instrument_emb)
        instrument_emb = self.fc_instrument_2(instrument_emb)
        instrument_emb = self.relu(instrument_emb)
        combined_emb = torch.cat((spectra_emb, instrument_emb), dim=-1)
        combined_emb = self.fc_combined(combined_emb)
        combined_emb = self.relu(combined_emb)
        output = self.fc_output(combined_emb)
        return output

    def step(self, batch):
        logits = self(batch)
        target = batch["labels"].int()
        loss = self.bce_loss(logits.view(-1, 1), target.view(-1, 1).float())
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).int()
        if target.numel() == 0 or preds.numel() == 0:
            raise ValueError("Empty predictions or targets in batch.")
        return loss, preds, target

    def training_step(self, batch, batch_idx):
        loss, preds, targets = self.step(batch)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.train_accuracy.update(preds, targets)
        self.train_precision.update(preds, targets)
        self.train_f1.update(preds, targets)
        self.train_recall.update(preds, targets)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, targets = self.step(batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.val_accuracy.update(preds, targets)
        self.val_precision.update(preds, targets)
        self.val_f1.update(preds, targets)
        self.val_recall.update(preds, targets)
        return {'loss': loss, 'logits': self(batch), 'labels': targets.view(-1, 1)}

    def on_train_epoch_end(self):
        self.log("train_accuracy", self.train_accuracy.compute(), prog_bar=True)
        self.log("train_precision", self.train_precision.compute())
        self.log("train_f1", self.train_f1.compute(), prog_bar=True)
        self.log("train_recall", self.train_recall.compute(), prog_bar=True)
        self.train_accuracy.reset()
        self.train_precision.reset()
        self.train_f1.reset()
        self.train_recall.reset()

    def on_validation_epoch_end(self):
        self.log("val_accuracy", self.val_accuracy.compute(), prog_bar=True)
        self.log("val_precision", self.val_precision.compute())
        self.log("val_f1", self.val_f1.compute(), prog_bar=True)
        self.log("val_recall", self.val_recall.compute(), prog_bar=True)
        self.val_accuracy.reset()
        self.val_precision.reset()
        self.val_f1.reset()
        self.val_recall.reset()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)


# Dataset Definition
class SimpleMassSpecDataset(Dataset):
    def __init__(self, mzml_file, csv_file, scaling="standardize"):
        self.mzml_file = mzml_file
        self.csv_file = csv_file
        self.scaling = scaling
        self.feature_stats = None
        self.excluded_columns = {"Scan"}
        self.scan_list = self.load_scans()
        self.ms2_data = self.load_ms2_data()
        if self.scaling == "standardize":
            self.compute_stats()
        self.data_pairs = self.align_scans()

    def get_scan_number(self, id_string):
        match = re.search(r'scan=(\d+)', id_string)
        return int(match.group(1)) if match else None

    def load_scans(self):
        scan_list = []
        with mzml.read(self.mzml_file) as reader:
            for spectrum in reader:
                ms_level = spectrum.get('ms level')
                id_string = spectrum.get('id')
                scan_number = self.get_scan_number(id_string)
                if scan_number is None or ms_level is None:
                    continue
                mz_array = spectrum.get('m/z array')
                intensity_array = spectrum.get('intensity array')
                if ms_level == 1:
                    mz_spectrum = sus.MsmsSpectrum(
                        identifier=str(scan_number),
                        precursor_mz=np.nan,
                        precursor_charge=np.nan,
                        mz=mz_array,
                        intensity=intensity_array,
                        retention_time=spectrum.get('scan start time', 0)
                    )
                    mz_spectrum = mz_spectrum.filter_intensity(min_intensity=0.01, max_num_peaks=400)
                    mz_spectrum = mz_spectrum.scale_intensity(scaling="root")
                scan_list.append({
                    'scan_number': scan_number,
                    'ms_level': ms_level,
                    'mz_array': mz_spectrum.mz,
                    'intensity_array': mz_spectrum.intensity
                })
        scan_list.sort(key=lambda x: x['scan_number'])
        return scan_list

    def load_ms2_data(self):
        ms2_df = pd.read_csv(self.csv_file)
        ms2_df['Scan'] = ms2_df['Scan'].astype(int)
        return ms2_df

    def get_instrument_settings_columns(self):
        return ["Scan", "RT [min]", "LowMass", "HighMass",
                "TIC", "BasePeakPosition", "BasePeakIntensity", "Charge State",
                "Monoisotopic M/Z", "Ion Injection Time (ms)",
                "Conversion Parameter C", "LM Search Window (ppm)", "Number of LM Found",
                "Mild Trapping Mode",
                "Energy1", "Orbitrap Resolution", "HCD Energy V(1)", "HCD Energy V(2)",
                "HCD Energy V(3)", "Number of Lock Masses", "LM m/z-Correction (ppm)"]

    def compute_stats(self):
        all_instrument_settings = []
        instrument_settings_cols = [
            col for col in self.get_instrument_settings_columns() if col not in self.excluded_columns
        ]
        for _, ms2_info in self.ms2_data.iterrows():
            instrument_settings = [
                float(ms2_info[col]) for col in instrument_settings_cols if col in ms2_info
            ]
            all_instrument_settings.append(instrument_settings)
        all_instrument_settings = np.array(all_instrument_settings)
        self.feature_stats = {
            i: {"mean": np.mean(all_instrument_settings[:, i]), "std": np.std(all_instrument_settings[:, i])}
            for i in range(all_instrument_settings.shape[1])
        }

    def scale_features(self, instrument_settings):
        scaled_settings = []
        stats_index = 0
        for col_index, value in enumerate(instrument_settings):
            if col_index in self.excluded_columns or col_index >= len(self.feature_stats):
                scaled_settings.append(value)
            else:
                mean_val = self.feature_stats[stats_index]["mean"]
                std_val = self.feature_stats[stats_index]["std"]
                scaled_value = (value - mean_val) / std_val if std_val > 0 else 0
                scaled_settings.append(scaled_value)
                stats_index += 1
        return np.array(scaled_settings, dtype=np.float32)

    def align_scans(self):
        data_pairs = []
        ms2_scan_info = self.ms2_data.set_index('Scan').to_dict('index')
        instrument_settings_cols = self.get_instrument_settings_columns()
        current_ms1_data = None
        current_ms1_scan_number = None
        for scan in self.scan_list:
            scan_number = scan['scan_number']
            ms_level = scan['ms_level']
            if ms_level == 1:
                current_ms1_data = {
                    'mz_array': scan['mz_array'],
                    'intensity_array': scan['intensity_array']
                }
                current_ms1_scan_number = scan_number
            elif ms_level == 2:
                if scan_number in ms2_scan_info and current_ms1_data is not None:
                    ms2_info = ms2_scan_info[scan_number]
                    instrument_settings = [float(ms2_info[col]) for col in instrument_settings_cols if
                                           col in ms2_info]
                    instrument_settings = self.scale_features(instrument_settings)
                    instrument_settings = np.array(instrument_settings, dtype=float)
                    selected_mass = ms2_info.get('SelectedMass1', None)
                    label = ms2_info['label']
                    data_pairs.append({
                        'ms1_scan_number': current_ms1_scan_number,
                        'ms2_scan_number': scan_number,
                        'mz_array': current_ms1_data['mz_array'],
                        'intensity_array': current_ms1_data['intensity_array'],
                        'instrument_settings': instrument_settings,
                        'precursor_mz': selected_mass,
                        'label': label
                    })
        return data_pairs

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        return self.data_pairs[idx]


# Collate Function
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


def predict_all(args):
    """Runs prediction on all samples and saves results to a CSV."""
    # Load model from checkpoint
    model = SimpleSpectraTransformer.load_from_checkpoint(args.checkpoint_path)
    model.eval()
    device = model.device
    # Read file paths from the input file
    with open(args.file_paths, 'r') as f:
        lines = f.readlines()
    mzml_files = [line.split(",")[0].strip() for line in lines]
    csv_files = [line.split(",")[1].strip() for line in lines]
    # Create dataset and dataloader
    dataset = SimpleMassSpecDataset(mzml_files[0], csv_files[0])
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn,
                            num_workers=args.num_workers)
    # Run predictions
    predictions = []
    probabilities = []
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            logits = model(batch)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).int()
            predictions.extend(preds.cpu().numpy())
            probabilities.extend(probs.cpu().numpy())
    # Save predictions
    results_df = pd.DataFrame(dataset.ms2_data)
    results_df['prediction'] = [p[0] for p in predictions]
    results_df['probability'] = [p[0] for p in probabilities]
    results_df.to_csv(args.output_csv, index=False)
    print(f"Predictions and probabilities saved to {args.output_csv}")


def select_low_prob_samples(args):
    """
    Performs a train/test split, predicts on training samples with label 0,
    and selects the indices of samples with the lowest prediction probabilities.
    """
    pl.seed_everything(1, workers=True)

    # Load model from checkpoint
    model = SimpleSpectraTransformer.load_from_checkpoint(args.checkpoint_path)
    model.eval()
    device = model.device
    print(f"Model is on device: {device}")
    # Load data
    with open(args.file_paths, 'r') as f:
        lines = f.readlines()
    mzml_files = [line.split(",")[0].strip() for line in lines]
    csv_files = [line.split(",")[1].strip() for line in lines]

    datasets = [SimpleMassSpecDataset(mzml, csv) for mzml, csv in zip(mzml_files, csv_files)]
    combined_dataset = ConcatDataset(datasets)

    # Perform train/test split to match the training process
    labels = [combined_dataset[i]['label'] for i in range(len(combined_dataset))]
    labels = np.array(labels, dtype=np.int64)
    all_indices = np.arange(len(combined_dataset))

    train_indices, _ = train_test_split(
        all_indices, test_size=0.3, random_state=1, stratify=labels
    )

    # Filter for training samples with label == 0
    train_indices_label_0 = [
        i for i in train_indices if combined_dataset[i]['label'] == 0
    ]

    # Create a new dataset and dataloader for these specific samples
    label_0_dataset = Subset(combined_dataset, train_indices_label_0)
    dataloader = DataLoader(
        label_0_dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        num_workers=args.num_workers
    )

    # Run predictions to get probabilities
    probabilities = []
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            logits = model(batch)
            probs = torch.sigmoid(logits)
            probabilities.extend(probs.cpu().numpy())

    # Associate probabilities with their original indices
    results = list(zip(train_indices_label_0, [p[0] for p in probabilities]))

    # Sort by probability (lowest first)
    results.sort(key=lambda x: x[1])

    # Select the top N samples with the lowest probability
    num_samples_to_select = int(args.num_samples)
    selected_samples = results[:num_samples_to_select]
    selected_indices = [item[0] for item in selected_samples]

    # Save the indices to a file
    output_file = "low_probability_indices_2.txt"
    with open(output_file, 'w') as f:
        for index in selected_indices:
            f.write(f"{index}\n")

    print(f"{len(selected_indices)} indices with the lowest probabilities saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="predict_all", choices=["predict_all", "select_low_prob"],
                        help="Operation mode")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--file_paths", type=str, required=True,
                        help="Path to the file containing mzML and CSV paths")
    parser.add_argument("--output_csv", type=str, default="predictions.csv", help="Path to save the predictions CSV")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of DataLoader workers")
    parser.add_argument("--num_samples", type=int, default=536270, help="Number of samples to select")
    args = parser.parse_args()

    if args.mode == "predict_all":
        predict_all(args)
    elif args.mode == "select_low_prob":
        select_low_prob_samples(args)