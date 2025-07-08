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
        # train_indices, val_indices, holdout_pos_indices are global indices for the combined_dataset
        self.train_indices = train_indices
        self.val_indices = val_indices  # These are global indices for the original validation set portion
        self.holdout_pos_indices = holdout_pos_indices  # These are global indices for holdout positives
        self.dataset = dataset  # This is the full combined_dataset
        self.validation_step_outputs = []  # Renamed from val_outputs for clarity with Lightning

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        # 'outputs' is the dictionary returned by the validation_step of the LightningModule
        if isinstance(outputs, dict) and 'logits' in outputs and 'labels' in outputs:
            # Detach and move to CPU to free GPU memory and prepare for aggregation
            self.validation_step_outputs.append({
                'logits': outputs['logits'].detach().cpu(),
                'labels': outputs['labels'].detach().cpu()
            })
        else:
            print("Warning: Validation step output in callback did not contain expected 'logits' and 'labels' keys.")

    def on_validation_epoch_end(self, trainer, pl_module):
        # Process outputs locally for the current rank. No all_gather.
        if not self.validation_step_outputs:
            rank_for_print = trainer.global_rank if hasattr(trainer, 'global_rank') else 'unknown'
            print(
                f"Warning (Rank {rank_for_print}): No validation_step_outputs collected. Estimating c and logging default PU metrics.")
            estimated_c = 1.0
            if len(self.holdout_pos_indices) > 0:
                try:
                    y_ph_prob = self.wrapper.predict_proba(self.holdout_pos_indices)[:, 1]
                    estimated_c = np.mean(y_ph_prob)
                    if estimated_c == 0:
                        estimated_c = 1e-8
                    elif estimated_c > 1:
                        estimated_c = 1.0
                except Exception as e:
                    print(
                        f"Warning (Rank {rank_for_print}): c-estimation failed during predict_proba: {e}. Using default c=1.0")
                    estimated_c = 1.0

            metrics_to_log = {
                'estimated_c_epoch': float(estimated_c), 'pu_val_auroc_gmm': 0.5, 'pu_val_auprc': 0.0,
                'pu_val_epr': 0.0, 'pu_val_accuracy': 0.0, 'pu_val_precision': 0.0,
                'pu_val_recall': 0.0, 'pu_val_f1': 0.0
            }
            if trainer.logger:
                trainer.logger.log_metrics(metrics_to_log, step=trainer.current_epoch)
            self.validation_step_outputs.clear()
            return

        # Concatenate collected logits and labels for THIS RANK
        # These are already detached and on CPU from on_validation_batch_end
        all_logits = torch.cat([out['logits'] for out in self.validation_step_outputs])
        all_labels = torch.cat([out['labels'] for out in self.validation_step_outputs])

        # .cpu().numpy() happens after sigmoid for val_probs_s1
        # .cpu().numpy() for val_true_labels (all_labels are already cpu tensors)
        val_probs_s1 = all_logits.sigmoid().numpy()
        val_true_labels = all_labels.numpy()

        self.validation_step_outputs.clear()

        # Estimate c using the fixed hold-out positives (runs on each rank)
        if len(self.holdout_pos_indices) == 0:
            rank_for_print = trainer.global_rank if hasattr(trainer, 'global_rank') else 'unknown'
            print(
                f"Warning (Rank {rank_for_print}): No hold-out positives for c estimation in callback. Using default c=1.0")
            estimated_c = 1.0
        else:
            try:
                y_ph_prob = self.wrapper.predict_proba(self.holdout_pos_indices)[:, 1]
                estimated_c = np.mean(y_ph_prob)
                if estimated_c == 0:
                    estimated_c = 1e-8
                elif estimated_c > 1:
                    estimated_c = 1.0
            except Exception as e:
                rank_for_print = trainer.global_rank if hasattr(trainer, 'global_rank') else 'unknown'
                print(
                    f"Warning (Rank {rank_for_print}): c-estimation failed during predict_proba: {e}. Using default c=1.0")
                estimated_c = 1.0

        # Ensure val_probs_s1 and val_true_labels are 1D arrays for metric calculations
        if val_probs_s1.ndim > 1 and val_probs_s1.shape[1] == 1:
            val_probs_s1 = val_probs_s1.squeeze(1)
        if val_true_labels.ndim > 1 and val_true_labels.shape[1] == 1:
            val_true_labels = val_true_labels.squeeze(1)

        if len(val_probs_s1) == 0 or len(val_true_labels) == 0:
            rank_for_print = trainer.global_rank if hasattr(trainer, 'global_rank') else 'unknown'
            print(
                f"Warning (Rank {rank_for_print}): After processing, validation probabilities or labels are empty. Skipping PU metrics calculation.")
            metrics_to_log = {
                'estimated_c_epoch': float(estimated_c), 'pu_val_auroc_gmm': 0.5, 'pu_val_auprc': 0.0,
                'pu_val_epr': 0.0, 'pu_val_accuracy': 0.0, 'pu_val_precision': 0.0,
                'pu_val_recall': 0.0, 'pu_val_f1': 0.0
            }
            if trainer.logger:
                trainer.logger.log_metrics(metrics_to_log, step=trainer.current_epoch)
            return

        if len(val_probs_s1) != len(val_true_labels):
            rank_for_print = trainer.global_rank if hasattr(trainer, 'global_rank') else 'unknown'
            print(
                f"FATAL ERROR in Callback (Rank {rank_for_print}): Length mismatch! Probs: {len(val_probs_s1)}, Labels: {len(val_true_labels)}. Skipping PU metrics.")
            return

        val_probs_y1_adjusted = np.clip(val_probs_s1 / estimated_c, 0, 1)
        val_predictions_pu_adjusted = np.where(val_probs_y1_adjusted > 0.5, 1.0, -1.0)

        val_true_labels_for_sklearn_metrics = val_true_labels
        val_predictions_for_sklearn_metrics = (val_probs_y1_adjusted > 0.5).astype(int)

        val_true_labels_for_pu_calc = np.array([1.0 if label == 1 else -1.0 for label in val_true_labels])
        val_labeled_pos_indices = np.where(val_true_labels == 1)[0]

        # Debug print (will print per rank)
        rank_for_print = trainer.global_rank if hasattr(trainer, 'global_rank') else 'unknown'
        # print(
        #     f"Debug (Rank {rank_for_print}): val_probs_y1_adjusted len: {len(val_probs_y1_adjusted)}, val_true_labels_for_pu_calc len: {len(val_true_labels_for_pu_calc)}, val_labeled_pos_indices len: {len(val_labeled_pos_indices)}")

        additional_pu_metrics = calculate_pu_metrics(
            probabilities=val_probs_y1_adjusted,
            true_labels=val_true_labels_for_pu_calc,
            labeled_pos_indices=val_labeled_pos_indices
        )

        # Standard metrics (accuracy, precision, recall, F1) calculated on 0/1 labels
        # Ensure val_true_labels_for_sklearn_metrics and val_predictions_for_sklearn_metrics are used here
        metrics = {
            'pu_val_accuracy': accuracy_score(val_true_labels_for_sklearn_metrics, val_predictions_for_sklearn_metrics),
            'pu_val_precision': precision_score(val_true_labels_for_sklearn_metrics,
                                                val_predictions_for_sklearn_metrics, average='binary',
                                                pos_label=1, zero_division=0),
            'pu_val_recall': recall_score(val_true_labels_for_sklearn_metrics, val_predictions_for_sklearn_metrics,
                                          average='binary',
                                          pos_label=1, zero_division=0),
            'pu_val_f1': f1_score(val_true_labels_for_sklearn_metrics, val_predictions_for_sklearn_metrics,
                                  average='binary', pos_label=1,
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
    print("\n--- calculate_pu_metrics --- DEBUG LOGS ---")
    print(f"Input probabilities (first 10): {probabilities[:10]}")
    print(f"Input true_labels (first 10): {true_labels[:10]}")
    print(f"Input labeled_pos_indices (first 10): {labeled_pos_indices[:10]}")
    print(
        f"Total samples: {len(probabilities)}, Total true labels: {len(true_labels)}, Total labeled_pos: {len(labeled_pos_indices)}")

    # Validate inputs
    if len(probabilities) != len(true_labels):
        raise ValueError(f"Length mismatch: probabilities ({len(probabilities)}) != true_labels ({len(true_labels)})")

    # Convert labels to binary format (0 and 1) for metric calculations
    binary_labels = np.array([1 if label == 1 else 0 for label in true_labels])
    print(f"Binary labels (first 10): {binary_labels[:10]}")

    # Validate labeled_pos_indices
    if not isinstance(labeled_pos_indices, np.ndarray):
        labeled_pos_indices = np.array(labeled_pos_indices)

    # Ensure indices are within bounds
    if len(labeled_pos_indices) > 0 and np.any(labeled_pos_indices >= len(probabilities)):
        print(f"Warning: Some labeled positive indices are out of bounds. Adjusting indices...")
        original_count = len(labeled_pos_indices)
        labeled_pos_indices = labeled_pos_indices[labeled_pos_indices < len(probabilities)]
        print(f"Adjusted labeled_pos_indices count from {original_count} to {len(labeled_pos_indices)}")

    auprc_val = 0.0
    epr_val = 0.0
    auroc_gmm_val = 0.5  # Default value

    if len(labeled_pos_indices) == 0:
        print("Warning: No valid labeled positive indices found after validation. Returning default metrics.")
        return {
            'auroc_gmm': 0.5,
            'auprc': 0.0,
            'epr': 0.0
        }

    # 1. Two-component Gaussian mixture model for AUROC
    print("\n-- GMM AUROC Calculation --")
    X = probabilities.reshape(-1, 1)

    try:
        gmm = GaussianMixture(n_components=2, random_state=SEED, reg_covar=1e-6)  # Added reg_covar for stability
        gmm.fit(X)

        means = gmm.means_.flatten()
        covariances = gmm.covariances_.flatten()
        weights = gmm.weights_.flatten()
        print(f"GMM Means: {means}")
        print(f"GMM Covariances: {covariances}")
        print(f"GMM Weights: {weights}")

        # Sort components by means to identify "positive" and "negative" components
        # This script's GMM AUROC uses roc_auc_score with GMM posteriors, not theoretical curves.
        sorted_idx = np.argsort(means)
        pos_component_idx = sorted_idx[1]  # Higher mean component is assumed positive
        neg_component_idx = sorted_idx[0]
        print(
            f"Positive component index (higher mean): {pos_component_idx}, Negative component index: {neg_component_idx}")

        # Calculate posterior probabilities for the identified positive component
        posteriors = gmm.predict_proba(X)[:, pos_component_idx]
        print(f"GMM Posteriors for positive component (first 10): {posteriors[:10]}")

        # Calculate AUROC using posteriors and binary labels
        if len(np.unique(binary_labels)) < 2:
            print("Warning: Only one class present in binary_labels. AUROC is not defined, defaulting to 0.5.")
            auroc_gmm_val = 0.5
        else:
            auroc_gmm_val = roc_auc_score(binary_labels, posteriors)
            print(f"Calculated AUROC GMM: {auroc_gmm_val}")

    except ValueError as e:
        print(f"ValueError during GMM fitting or AUROC calculation: {e}. Defaulting AUROC GMM to 0.5.")
        auroc_gmm_val = 0.5
    except Exception as e:
        print(f"Unexpected error during GMM AUROC calculation: {e}. Defaulting AUROC GMM to 0.5.")
        auroc_gmm_val = 0.5

    # 2. Percentile rank and EPR calculation
    print("\n-- Percentile Rank & EPR Calculation --")
    # Calculate percentile ranks for all samples
    # Ensure there's more than one rank to avoid division by zero if all probs are same
    if len(probabilities) > 1 and np.max(probabilities) != np.min(probabilities):
        ranks = rankdata(probabilities, method='average')
        percentile_ranks = (ranks - 1) / (len(ranks) - 1)
    else:  # Handle cases with single sample or all probabilities being identical
        percentile_ranks = np.zeros_like(probabilities) if len(probabilities) > 0 else np.array([])

    print(f"Percentile ranks (first 10): {percentile_ranks[:10]}")

    # Calculate percentile ranks for labeled positives
    labeled_pos_ranks = percentile_ranks[labeled_pos_indices]
    print(f"Labeled positive ranks (all): {labeled_pos_ranks}")

    # Calculate area under percentile rank curve (AUPRC)
    if len(labeled_pos_ranks) > 0:
        sorted_ranks = np.sort(labeled_pos_ranks)
        n_pos = len(sorted_ranks)
        print(f"Sorted labeled positive ranks (for AUPRC): {sorted_ranks}")
        if n_pos > 1:
            x_values_for_auprc = np.arange(1, n_pos + 1) / n_pos
            print(f"X-values for AUPRC: {x_values_for_auprc}")
            auprc_val = np.trapz(sorted_ranks, x=x_values_for_auprc)
        elif n_pos == 1:  # If only one positive, AUPRC is its rank (or 0 if we consider area)
            auprc_val = float(sorted_ranks[0])  # Or interpret as you see fit for single point
        # If n_pos is 0, auprc_val remains 0.0 as initialized
        print(f"Calculated AUPRC: {auprc_val}")
    else:
        print("No labeled positive ranks to calculate AUPRC.")

    # Calculate EPR (proportion of labeled positives in top k%)
    k = 0.1  # Top 10%
    threshold_epr = 1 - k
    print(f"EPR k: {k}, EPR threshold (percentile > {threshold_epr})")
    if len(labeled_pos_ranks) > 0:
        epr_val = np.mean(labeled_pos_ranks > threshold_epr)
        print(f"EPR (mean of labeled_pos_ranks > {threshold_epr}): {epr_val}")
    else:
        print("No labeled positive ranks to calculate EPR.")

    print("--- End calculate_pu_metrics DEBUG LOGS ---")
    return {
        'auroc_gmm': float(auroc_gmm_val),
        'auprc': float(auprc_val),
        'epr': float(epr_val)
    }


def main(args):
    # Print GPU information
    if torch.cuda.is_available():
        print(f"Available GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    experiment_name = "pu_learning_fixed_holdout_train_val_only"  # Updated experiment name
    log_dir = os.path.join(args.log_dir, experiment_name)
    os.makedirs(log_dir, exist_ok=True)
    csv_logger = CSVLogger(
        save_dir=args.log_dir,
        name=experiment_name,
        version=datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),  # Unique version
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

    # Split: train/val using stratified split (70% train, 30% val)
    train_indices, val_indices = train_test_split(
        indices,  # Use all indices
        test_size=0.3,  # 30% for validation
        random_state=SEED,
        stratify=labels  # Stratify on all labels
    )

    # Verify positive samples in each set
    train_labels = labels[train_indices]
    val_labels = labels[val_indices]

    print("\nSplit Statistics:")
    print(f"Training set: {len(train_indices)} samples, {np.sum(train_labels == 1)} positives")
    print(f"Validation set: {len(val_indices)} samples, {np.sum(val_labels == 1)} positives")

    # --- Fixed hold-out logic (applied to the 70% training set) ---
    positive_train_indices = train_indices[train_labels == 1]
    hold_out_ratio = args.hold_out_ratio

    if len(positive_train_indices) == 0:
        print("Warning: No positive samples in the initial training split to create a hold-out set.")
        holdout_pos_indices = np.array([], dtype=int)
        train_indices_no_holdout = train_indices.copy()  # Use original train_indices
    else:
        n_hold_out = max(1, int(np.ceil(len(positive_train_indices) * hold_out_ratio)))
        if n_hold_out >= len(positive_train_indices):
            n_hold_out = len(positive_train_indices) - 1 if len(positive_train_indices) > 1 else 0
            if n_hold_out < 1 and len(positive_train_indices) > 1:
                print(
                    "Warning: Adjusted n_hold_out to be less than total positives in training. Hold-out set might be very small or empty.")

        if n_hold_out == 0:
            print(
                "Warning: Calculated n_hold_out is 0. No samples for hold-out set from training positives. C-estimation might be impacted.")
            holdout_pos_indices = np.array([], dtype=int)
            train_indices_no_holdout = train_indices.copy()
        else:
            # Split hold-out positives from the training positives
            # Ensure positive_train_indices is not empty before splitting
            if len(positive_train_indices) > n_hold_out:
                _, holdout_pos_indices = train_test_split(
                    positive_train_indices,
                    test_size=n_hold_out,
                    random_state=SEED
                    # No stratification needed as we are splitting from positives only
                )
                train_indices_no_holdout = np.setdiff1d(train_indices, holdout_pos_indices)
            elif len(
                    positive_train_indices) > 0:  # if positive_train_indices has elements but not enough for n_hold_out
                print(
                    f"Warning: Not enough positive samples ({len(positive_train_indices)}) for the desired hold_out_ratio resulting in n_hold_out={n_hold_out}. Using all available positives for holdout.")
                holdout_pos_indices = positive_train_indices.copy()
                train_indices_no_holdout = np.setdiff1d(train_indices, holdout_pos_indices)
            else:  # Should have been caught by len(positive_train_indices) == 0 earlier
                holdout_pos_indices = np.array([], dtype=int)
                train_indices_no_holdout = train_indices.copy()

    print(f"\nHold-out set: {len(holdout_pos_indices)} positive samples (from training set)")
    print(f"Training set after hold-out: {len(train_indices_no_holdout)} samples")

    train_labels_no_holdout = labels[train_indices_no_holdout]  # Get labels for the final training set
    if np.sum(train_labels_no_holdout == 1) == 0 and len(train_indices_no_holdout) > 0:
        print(
            "Warning: No positive samples left in training set after hold-out! This may be intended if all positives are in holdout, or training set became empty of positives.")

    # Model initialization
    model = SimpleSpectraTransformer(
        d_model=args.d_model,
        n_layers=args.n_layers,
        dropout=args.dropout,
        lr=args.lr,
        instrument_embedding_dim=args.instrument_embedding_dim
    )

    wrapper_args = dict(
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
        logger=csv_logger
    )

    try:
        pytorch_model = ImprovedPyTorchSklearnWrapper(**wrapper_args, force_cpu=args.force_cpu)
    except Exception as e:
        print(f"Error during model initialization with preferred device: {e}")
        print("Trying to use CPU only for all operations...")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        model_cpu = SimpleSpectraTransformer(
            d_model=args.d_model, n_layers=args.n_layers, dropout=args.dropout,
            lr=args.lr, instrument_embedding_dim=args.instrument_embedding_dim
        )
        pytorch_model = ImprovedPyTorchSklearnWrapper(**wrapper_args, model=model_cpu, force_cpu=True)

    # Callback for per-epoch PU metrics
    pu_callback = FixedHoldoutPUCallback(
        wrapper=pytorch_model,
        train_indices=train_indices_no_holdout,
        val_indices=val_indices,
        holdout_pos_indices=holdout_pos_indices,  # These are from the training data
        dataset=combined_dataset
    )

    # Train the model using the wrapper's fit method with validation data and callback
    _, trainer = pytorch_model.fit(
        train_indices=train_indices_no_holdout,
        val_indices=val_indices,
        callbacks=[pu_callback]
    )

    # --- After training: Final summary ---
    # No test set evaluation. The validation metrics are logged by the callback.

    final_metrics_summary = {}
    # Optionally, re-calculate and log final validation metrics here if needed,
    # or rely on the last epoch's logged validation metrics.
    # For simplicity, we will just save the model and hyperparameters.

    print("\nTraining finished.")

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
        # Add last epoch's validation metrics to summary if desired
        # last_val_metrics = trainer.logged_metrics # Access logged metrics
        # f.write("\nLast Validation Metrics (from CSVLogger):\n")
        # for metric, value in last_val_metrics.items():
        #    if 'pu_val' in metric or 'estimated_c_epoch' in metric : # Log relevant validation metrics
        #        f.write(f"{metric}: {value:.4f}\n")

    print(f"\nModel saved to {model_save_path}")
    print(f"Experiment summary saved to {summary_path}")
    print(f"Logs can be found in: {log_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_paths", type=str, required=True, help="Path to file containing mzML and CSV paths")
    parser.add_argument("--log_dir", type=str, default="./logs", help="Directory for logs")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of DataLoader workers")
    parser.add_argument("--d_model", type=int, default=64, help="Model dimension")
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