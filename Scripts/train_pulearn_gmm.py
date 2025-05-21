import os
import datetime

# Enable MPS fallback for Mac
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
# Force CPU usage

import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn as nn
import lightning.pytorch as pl
from torch.utils.data import DataLoader, Subset, ConcatDataset
import numpy as np
import argparse
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import Callback
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.mixture import GaussianMixture
from scipy.stats import rankdata
from src.transformers.simpleDataset import SimpleMassSpecDataset
from src.transformers.model_simplified import SimpleSpectraTransformer
from lightning.pytorch.strategies import DDPStrategy

SEED = 1
pl.seed_everything(SEED, workers=True)


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


class ImprovedPyTorchSklearnWrapper:
    def __init__(self, model, dataset, d_model=64, n_layers=2, dropout=0.3, lr=0.001, batch_size=64, epochs=5,
                 device=None, num_workers=4, instrument_embedding_dim=32, force_cpu=False, logger=None):
        self.model = model
        self.dataset = dataset
        self.d_model = d_model
        self.n_layers = n_layers
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.num_workers = num_workers
        self.instrument_embedding_dim = instrument_embedding_dim
        self.force_cpu = force_cpu
        self.logger = logger

        # Determine device
        if force_cpu:
            self.device = torch.device("cpu")
        elif device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else
                                       "mps" if torch.backends.mps.is_available() else
                                       "cpu")
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")

        # Move model to the appropriate device
        self.model = self.model.to(self.device)

        # Ensure all model components are on the same device
        if hasattr(self.model, 'spectrum_encoder'):
            self.model.spectrum_encoder = self.model.spectrum_encoder.to(self.device)
            if hasattr(self.model.spectrum_encoder, 'peak_encoder'):
                self.model.spectrum_encoder.peak_encoder = self.model.spectrum_encoder.peak_encoder.to(self.device)
                if hasattr(self.model.spectrum_encoder.peak_encoder, 'mz_encoder'):
                    self.model.spectrum_encoder.peak_encoder.mz_encoder = self.model.spectrum_encoder.peak_encoder.mz_encoder.to(
                        self.device)

        self.is_fitted_ = False
        self.classes_ = np.array([-1.0, 1.0])

    def predict_proba(self, indices):
        self.model.eval()
        test_dataset = Subset(self.dataset, indices)
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            persistent_workers=True,
            pin_memory=True
        )
        all_probs = []
        with torch.no_grad():
            for batch in test_loader:
                # Move all tensors to the model's device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                # Ensure model is on the correct device
                self.model = self.model.to(self.device)
                # Forward pass
                logits = self.model(batch)
                probs = torch.sigmoid(logits)
                # Move probabilities to CPU before converting to numpy
                all_probs.append(probs.cpu().numpy())
        probs_numpy = np.vstack(all_probs)
        return np.column_stack([1 - probs_numpy, probs_numpy])

    def fit(self, train_indices, val_indices=None, callbacks=None):
        """
        Fits the model on the training data and optionally validates on validation data.

        Args:
            train_indices: Indices for training data
            val_indices: Optional indices for validation data
            callbacks: Optional list of PyTorch Lightning callbacks
        """
        train_dataset = Subset(self.dataset, train_indices)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            persistent_workers=True,
            pin_memory=True
        )

        val_loader = None
        if val_indices is not None:
            val_dataset = Subset(self.dataset, val_indices)
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                collate_fn=collate_fn,
                persistent_workers=True,
                pin_memory=True,
                shuffle=False
            )

        if callbacks is None:
            callbacks = []

        trainer = pl.Trainer(
            max_epochs=self.epochs,
            logger=self.logger,
            callbacks=callbacks,
            enable_progress_bar=True,
            accelerator='gpu',
            devices=2,
            strategy=DDPStrategy(gradient_as_bucket_view=True, static_graph=True),
            precision='16-mixed'  # Use mixed precision for better performance
        )
        trainer.fit(self.model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        self.is_fitted_ = True
        return self, trainer

    def predict(self, indices, threshold=0.5):
        probs = self.predict_proba(indices)[:, 1]
        return np.array([1.0 if p > threshold else -1.0 for p in probs])


class FixedHoldoutPUCallback(Callback):
    def __init__(self, wrapper, train_indices, val_indices, holdout_pos_indices, dataset):
        super().__init__()
        self.wrapper = wrapper
        self.train_indices = train_indices
        self.val_indices = val_indices
        self.holdout_pos_indices = holdout_pos_indices
        self.dataset = dataset
        self.val_outputs = []
        self.val_batch_indices = []  # Store batch indices to track which samples we have

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Store the logits from the validation step output
        if isinstance(outputs, dict) and 'logits' in outputs:
            self.val_outputs.append(outputs['logits'])
            # Store the indices for this batch
            start_idx = batch_idx * trainer.val_dataloaders.batch_size
            end_idx = min(start_idx + trainer.val_dataloaders.batch_size, len(self.val_indices))
            self.val_batch_indices.extend(range(start_idx, end_idx))
        else:
            print("Warning: Validation step did not return logits in expected format")

    def on_validation_epoch_end(self, trainer, pl_module):
        # Estimate c using the fixed hold-out positives
        if len(self.holdout_pos_indices) == 0:
            print("Warning: No hold-out positives for c estimation.")
            estimated_c = 1.0
        else:
            X_ph_reshaped = np.array(self.holdout_pos_indices).reshape(-1, 1)
            y_ph_prob = self.wrapper.predict_proba(self.holdout_pos_indices)[:, 1]
            estimated_c = np.mean(y_ph_prob)
            if estimated_c == 0:
                estimated_c = 1e-8
            elif estimated_c > 1:
                estimated_c = 1.0

        # Get validation probabilities from stored outputs
        if self.val_outputs:
            val_probs_s1 = torch.cat(self.val_outputs).sigmoid().cpu().numpy()
            # Get true labels only for the samples we have predictions for
            val_true_labels = np.array([self.dataset[self.val_indices[i]]['label'] for i in self.val_batch_indices])
        else:
            print("Warning: No validation outputs collected, falling back to wrapper prediction")
            val_probs_s1 = self.wrapper.predict_proba(self.val_indices)[:, 1]
            val_true_labels = np.array([self.dataset[i]['label'] for i in self.val_indices])

        val_probs_y1_adjusted = np.clip(val_probs_s1 / estimated_c, 0, 1)
        val_predictions_pu_adjusted = np.array([1.0 if p > 0.5 else -1.0 for p in val_probs_y1_adjusted])
        val_true_labels_pu = np.array([1.0 if label == 1 else -1.0 for label in val_true_labels])

        # Calculate additional PU metrics for validation set
        val_labeled_pos_indices = np.where(val_true_labels == 1)[0]
        additional_pu_metrics = calculate_pu_metrics(
            val_probs_y1_adjusted,
            val_true_labels_pu,
            val_labeled_pos_indices
        )

        # Combine all metrics
        metrics = {
            'pu_val_accuracy': accuracy_score(val_true_labels_pu, val_predictions_pu_adjusted),
            'pu_val_precision': precision_score(val_true_labels_pu, val_predictions_pu_adjusted, average='binary',
                                                pos_label=1.0, zero_division=0),
            'pu_val_recall': recall_score(val_true_labels_pu, val_predictions_pu_adjusted, average='binary',
                                          pos_label=1.0, zero_division=0),
            'pu_val_f1': f1_score(val_true_labels_pu, val_predictions_pu_adjusted, average='binary', pos_label=1.0,
                                  zero_division=0),
            'estimated_c_epoch': float(estimated_c),
            'pu_val_auroc_gmm': additional_pu_metrics['auroc_gmm'],
            'pu_val_auprc': additional_pu_metrics['auprc'],
            'pu_val_epr': additional_pu_metrics['epr']
        }

        # Log metrics
        trainer.logger.log_metrics(metrics, step=trainer.current_epoch)

        # Print metrics
        print(f"\nEpoch {trainer.current_epoch} PU Metrics (Fixed Holdout):")
        for name, value in metrics.items():
            print(f"  {name}: {value:.4f}")

        # Clear validation outputs for next epoch
        self.val_outputs = []
        self.val_batch_indices = []


def calculate_pu_metrics(probabilities, true_labels, labeled_pos_indices):
    """
    Calculate additional PU learning metrics:
    1. AUROC using two-component Gaussian mixture model
    2. Percentile rank of labeled positives and EPR

    Args:
        probabilities: Array of probability scores for all samples (between 0 and 1)
        true_labels: Array of true labels (1 for positive, -1 for unlabeled)
        labeled_pos_indices: Indices of known positive samples

    Returns:
        Dictionary containing the calculated metrics
    """
    # Validate inputs
    if len(probabilities) != len(true_labels):
        raise ValueError(f"Length mismatch: probabilities ({len(probabilities)}) != true_labels ({len(true_labels)})")

    # Convert labels to binary format (0 and 1) for metric calculations
    binary_labels = np.array([1 if label == 1 else 0 for label in true_labels])

    # Validate labeled_pos_indices
    if not isinstance(labeled_pos_indices, np.ndarray):
        labeled_pos_indices = np.array(labeled_pos_indices)

    # Ensure indices are within bounds
    if np.any(labeled_pos_indices >= len(probabilities)):
        print(f"Warning: Some labeled positive indices are out of bounds. Adjusting indices...")
        labeled_pos_indices = labeled_pos_indices[labeled_pos_indices < len(probabilities)]

    if len(labeled_pos_indices) == 0:
        print("Warning: No valid labeled positive indices found. Returning default metrics.")
        return {
            'auroc_gmm': 0.5,
            'auprc': 0.0,
            'epr': 0.0
        }

    # 1. Two-component Gaussian mixture model for AUROC
    # Reshape probabilities for GMM
    X = probabilities.reshape(-1, 1)

    # Fit two-component GMM
    gmm = GaussianMixture(n_components=2, random_state=SEED)
    gmm.fit(X)

    # Get component means and sort them
    means = gmm.means_.flatten()
    sorted_idx = np.argsort(means)
    pos_component = sorted_idx[1]  # Higher mean component

    # Calculate posterior probabilities for positive component
    posteriors = gmm.predict_proba(X)[:, pos_component]

    # Calculate AUROC using posteriors and binary labels
    try:
        auroc = roc_auc_score(binary_labels, posteriors)
    except ValueError:
        print("Warning: ROC calculation failed. Using default value.")
        auroc = 0.5

    # 2. Percentile rank and EPR calculation
    # Calculate percentile ranks for all samples
    ranks = rankdata(probabilities, method='average')
    percentile_ranks = (ranks - 1) / (len(ranks) - 1)

    # Calculate percentile ranks for labeled positives
    labeled_pos_ranks = percentile_ranks[labeled_pos_indices]

    # Calculate area under percentile rank curve
    # Sort labeled positive ranks
    sorted_ranks = np.sort(labeled_pos_ranks)
    n_pos = len(sorted_ranks)

    # Calculate Area under Percentile Rank Curve using trapezoidal rule
    auprc = np.trapz(np.arange(1, n_pos + 1) / n_pos, sorted_ranks)

    # Calculate EPR (proportion of labeled positives in top k%)
    k = 0.1  # Top 10%
    threshold = 1 - k
    epr = np.mean(labeled_pos_ranks > threshold)

    return {
        'auroc_gmm': float(auroc),
        'auprc': float(auprc),
        'epr': float(epr)
    }


def main(args):
    # Print GPU information
    if torch.cuda.is_available():
        print(f"Available GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    experiment_name = "pu_learning_fixed_holdout_20_05"  # New experiment name
    log_dir = os.path.join(args.log_dir, experiment_name)
    os.makedirs(log_dir, exist_ok=True)
    csv_logger = CSVLogger(
        save_dir=args.log_dir,
        name=experiment_name,
        version="version_2",
        flush_logs_every_n_steps=10
    )
    # Read file paths
    with open(args.file_paths, 'r') as f:
        lines = f.readlines()
    mzml_files = [line.split(",")[0].strip() for line in lines]
    csv_files = [line.split(",")[1].strip() for line in lines]
    datasets = [SimpleMassSpecDataset(mzml, csv) for mzml, csv in zip(mzml_files, csv_files)]
    combined_dataset = ConcatDataset(datasets)
    labels = [combined_dataset[i]['label'] for i in range(len(combined_dataset))]
    labels = np.array(labels, dtype=np.int64)
    indices = np.arange(len(combined_dataset))

    # Print dataset statistics
    print(f"Total samples: {len(indices)}")
    print(f"Positive samples: {np.sum(labels == 1)}")
    print(f"Unlabeled samples: {np.sum(labels == 0)}")

    if np.sum(labels == 1) == 0:
        raise ValueError("No positive samples found in the dataset!")

    # First split: test set using stratified split
    train_val_indices, test_indices = train_test_split(
        indices,
        test_size=0.2,
        random_state=SEED,
        stratify=labels
    )

    # Get labels for train_val set
    train_val_labels = labels[train_val_indices]

    # Second split: train/val using stratified split
    train_indices, val_indices = train_test_split(
        train_val_indices,
        test_size=0.2,
        random_state=SEED,
        stratify=train_val_labels
    )

    # Verify positive samples in each set
    train_labels = labels[train_indices]
    val_labels = labels[val_indices]
    test_labels = labels[test_indices]

    print("\nSplit Statistics:")
    print(f"Training set: {len(train_indices)} samples, {np.sum(train_labels == 1)} positives")
    print(f"Validation set: {len(val_indices)} samples, {np.sum(val_labels == 1)} positives")
    print(f"Test set: {len(test_indices)} samples, {np.sum(test_labels == 1)} positives")

    # --- Fixed hold-out logic ---
    positive_train_indices = train_indices[train_labels == 1]
    hold_out_ratio = args.hold_out_ratio
    n_hold_out = max(1, int(np.ceil(len(positive_train_indices) * hold_out_ratio)))

    if n_hold_out >= len(positive_train_indices):
        n_hold_out = len(positive_train_indices) - 1
        if n_hold_out < 1:
            raise ValueError("Not enough positive samples in training set for hold-out!")

    # Split hold-out positives - no need for stratification here since we're only splitting positives
    _, holdout_pos_indices = train_test_split(
        positive_train_indices,
        test_size=n_hold_out,
        random_state=SEED
    )

    # Remove hold-out from training set
    train_indices_no_holdout = np.setdiff1d(train_indices, holdout_pos_indices)

    print(f"\nHold-out set: {len(holdout_pos_indices)} positive samples")
    print(f"Training set after hold-out: {len(train_indices_no_holdout)} samples")

    # Verify we still have positives in training set after hold-out
    train_labels_no_holdout = labels[train_indices_no_holdout]
    if np.sum(train_labels_no_holdout == 1) == 0:
        raise ValueError("No positive samples left in training set after hold-out!")

    # Model initialization
    model = SimpleSpectraTransformer(
        d_model=args.d_model,
        n_layers=args.n_layers,
        dropout=args.dropout,
        lr=args.lr,
        instrument_embedding_dim=args.instrument_embedding_dim
    )

    try:
        pytorch_model = ImprovedPyTorchSklearnWrapper(
            model=model,
            dataset=combined_dataset,
            d_model=args.d_model,
            n_layers=args.n_layers,
            dropout=args.dropout,
            lr=args.lr,
            batch_size=args.batch_size,
            epochs=args.epochs,
            num_workers=args.num_workers,
            instrument_embedding_dim=args.instrument_embedding_dim,
            force_cpu=args.force_cpu,
            logger=csv_logger
        )
    except Exception as e:
        print(f"Error during model initialization: {e}")
        print("Trying to use CPU only for all operations...")
        # Force CPU usage for all operations
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["PYTORCH_MPS_DEVICE"] = "0"

        # Create a new model on CPU
        model = SimpleSpectraTransformer(
            d_model=args.d_model,
            n_layers=args.n_layers,
            dropout=args.dropout,
            lr=args.lr,
            instrument_embedding_dim=args.instrument_embedding_dim
        )

        pytorch_model = ImprovedPyTorchSklearnWrapper(
            model=model,
            dataset=combined_dataset,
            d_model=args.d_model,
            n_layers=args.n_layers,
            dropout=args.dropout,
            lr=args.lr,
            batch_size=args.batch_size,
            epochs=args.epochs,
            num_workers=args.num_workers,
            instrument_embedding_dim=args.instrument_embedding_dim,
            force_cpu=True,
            logger=csv_logger
        )

    # Callback for per-epoch PU metrics
    pu_callback = FixedHoldoutPUCallback(
        wrapper=pytorch_model,
        train_indices=train_indices_no_holdout,
        val_indices=val_indices,
        holdout_pos_indices=holdout_pos_indices,
        dataset=combined_dataset
    )

    # Train the model using the wrapper's fit method with validation data and callback
    _, trainer = pytorch_model.fit(
        train_indices=train_indices_no_holdout,
        val_indices=val_indices,
        callbacks=[pu_callback]
    )

    # --- After training: PU metrics on test set ---

    y_ph_prob = pytorch_model.predict_proba(holdout_pos_indices)[:, 1]
    estimated_c = np.mean(y_ph_prob) if len(y_ph_prob) > 0 else 1.0
    if estimated_c == 0:
        estimated_c = 1e-8
    elif estimated_c > 1:
        estimated_c = 1.0
    X_test_indices = np.array(test_indices).reshape(-1, 1)
    test_probs_s1 = pytorch_model.predict_proba(test_indices)[:, 1]
    test_probs_y1_adjusted = np.clip(test_probs_s1 / estimated_c, 0, 1)
    test_predictions_pu_adjusted = np.array([1.0 if p > 0.5 else -1.0 for p in test_probs_y1_adjusted])
    test_true_labels = np.array([combined_dataset[i]['label'] for i in test_indices])
    test_true_labels_pu = np.array([1.0 if label == 1 else -1.0 for label in test_true_labels])

    # Calculate additional PU metrics
    test_labeled_pos_indices = np.where(test_true_labels == 1)[0]
    additional_pu_metrics = calculate_pu_metrics(
        test_probs_y1_adjusted,
        test_true_labels_pu,
        test_labeled_pos_indices
    )

    pu_test_metrics = {
        'pu_test_accuracy': accuracy_score(test_true_labels_pu, test_predictions_pu_adjusted),
        'pu_test_precision': precision_score(test_true_labels_pu, test_predictions_pu_adjusted, average='binary',
                                             pos_label=1.0, zero_division=0),
        'pu_test_recall': recall_score(test_true_labels_pu, test_predictions_pu_adjusted, average='binary',
                                       pos_label=1.0, zero_division=0),
        'pu_test_f1': f1_score(test_true_labels_pu, test_predictions_pu_adjusted, average='binary', pos_label=1.0,
                               zero_division=0),
        'estimated_c_test': float(estimated_c),
        **additional_pu_metrics
    }

    print("\nPU-Adjusted Test Set Metrics (Fixed Holdout):")
    for metric_name, value in pu_test_metrics.items():
        print(f"{metric_name}: {value:.4f}")
    csv_logger.log_metrics(pu_test_metrics)

    # Save model and summary
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_path = os.path.join(log_dir, f"pu_model_{timestamp}.pt")
    torch.save(pytorch_model.model.state_dict(), model_save_path)
    summary_path = os.path.join(log_dir, f"experiment_summary_{timestamp}.txt")
    with open(summary_path, 'w') as f:
        f.write("=== Experiment Summary ===\n\n")
        f.write("Hyperparameters:\n")
        for param, value in vars(args).items():
            f.write(f"{param}: {value}\n")
        f.write("\nPU-Adjusted Test Metrics (Fixed Holdout):\n")
        for metric, value in pu_test_metrics.items():
            f.write(f"{metric}: {value:.4f}\n")
    print(f"\nExperiment summary saved to {summary_path}")
    print(f"Logs can be found in: {log_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_paths", type=str, required=True, help="Path to file containing mzML and CSV paths")
    parser.add_argument("--log_dir", type=str, default="./logs", help="Directory for logs")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of DataLoader workers")
    parser.add_argument("--d_model", type=int, default=256, help="Model dimension")
    parser.add_argument("--n_layers", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--dropout", type=float, default=0.135378967114, help="Dropout rate")
    parser.add_argument("--lr", type=float, default=0.0002269876583, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--hold_out_ratio", type=float, default=0.1, help="Hold-out ratio for PU learning")
    parser.add_argument("--instrument_embedding_dim", type=int, default=16,
                        help="Dimension of the instrument embedding output")
    parser.add_argument("--force_cpu", action="store_true", help="Force using CPU instead of GPU/MPS")
    args = parser.parse_args()
    main(args)