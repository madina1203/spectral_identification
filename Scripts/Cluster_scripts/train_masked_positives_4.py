import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import sys

# --- Set up logging ---
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='debug_self_training_14_07.log',
                    filemode='w')
# ---

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
print("Current sys.path:")
print("\n".join(sys.path))
import torch
import torch.nn as nn
import lightning.pytorch as pl
from torch.utils.data import DataLoader, Subset, ConcatDataset
import numpy as np
import argparse
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from torch.nn.utils.rnn import pad_sequence

from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, auc, \
    precision_recall_curve
from sklearn.mixture import GaussianMixture
from scipy.stats import norm, rankdata
import matplotlib.pyplot as plt
import re
import csv

from src.transformers.simpleDataset_3 import SimpleMassSpecDataset
from src.transformers.model_simplified_2 import SimpleSpectraTransformer

SEED = 1
pl.seed_everything(SEED, workers=True)


# A custom Dataset wrapper to use updated labels from SelfTrainingClassifier
class DynamicLabelDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, indices, labels):
        self.base_dataset = base_dataset
        # Create a map from the original index to the new label
        self.label_map = {index: label for index, label in zip(indices, labels)}
        # We will only train on samples that have been assigned a label (i.e., not -1)
        self.active_indices = [idx for idx in indices if self.label_map[idx] != -1]

    def __len__(self):
        return len(self.active_indices)

    def __getitem__(self, idx):
        # The index for the original dataset
        original_idx = self.active_indices[idx]
        # Get the original data item
        item = self.base_dataset[original_idx]
        # IMPORTANT: Override the label with the new one from the y vector
        item['label'] = self.label_map[original_idx]
        return item


# Define a new callback to suppress per-step metric logging from the model
class SuppressModelLoggingCallback(pl.Callback):
    """
    A callback to prevent the model from logging metrics to the logger.
    Metrics will still be available in `trainer.callback_metrics` for other callbacks to use.
    """

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Prevent train step metrics from being written to the logger by removing them from the dict.
        # The trainer.logged_metrics property returns a mutable dict, so we can modify it in-place.
        keys_to_remove = [k for k in trainer.logged_metrics if k.startswith("train_")]
        for key in keys_to_remove:
            del trainer.logged_metrics[key]

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        # Prevent validation step metrics from being written to the logger by removing them from the dict.
        keys_to_remove = [k for k in trainer.logged_metrics if k.startswith("val_")]
        for key in keys_to_remove:
            del trainer.logged_metrics[key]


# Define a custom collate function to handle variable-length sequences
def collate_fn(batch):
    """
    Custom collate function for handling variable-length spectra data
    """
    mz_arrays = [torch.tensor(item['mz_array'], dtype=torch.float32) for item in batch]
    intensity_arrays = [torch.tensor(item['intensity_array'], dtype=torch.float32) for item in batch]

    # Convert list of instrument settings arrays to a single NumPy array first
    instrument_settings = np.array([item['instrument_settings'] for item in batch], dtype=np.float32)
    # Then convert the NumPy array to a PyTorch tensor
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


# wrapper for pytorch model compatible with scikit learn estimator
class ImprovedPyTorchSklearnWrapper(BaseEstimator, ClassifierMixin):
    """
    Wrapper for pytorch lightning
    """

    def __init__(self, model=None, dataset=None, d_model=64, n_layers=2,
                 dropout=0.3, lr=0.001, batch_size=64, epochs=5, device_str=None, num_workers=4,
                 instrument_embedding_dim=32, force_cpu=False, csv_logger=None, val_indices=None,
                 val_true_labels_pu=None, combined_dataset=None, log_dir=None, validation_threshold=0.95,
                 masked_positive_indices=None, stc_threshold=0.95, initial_unlabeled_indices_absolute=None):
        """
        Initialization of wrapper.

        Parameters:
        -----------
        model : PyTorch model or None
                    If None, a new SimpleSpectraTransformer model is created.
                dataset : PyTorch Dataset
                    Dataset for training.
                val_dataset : PyTorch Dataset or None
                    Dataset for validation.
                d_model : int
                    Model dimension.
                n_layers : int
                    Number of layers.
                dropout : float
                    Dropout probability.
                lr : float
                    Learning rate.
                batch_size : int
                    Batch size.
                epochs : int
                    Number of epochs.
                device_str : str or None
                    Computing device ('cpu', 'cuda', 'mps').
                num_workers : int
                    Number of worker processes for data loading.
                instrument_embedding_dim : int
                    Dimension of the instrument embedding.
        """
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
        self.validation_threshold = validation_threshold

        self.force_cpu = force_cpu
        self.device_str = device_str
        self.csv_logger = csv_logger

        self._actual_torch_device_obj = None

        self.is_fitted_ = False
        self.classes_ = np.array([-1.0, 1.0])

        # Initialize validation data attributes
        self.val_indices = val_indices
        self.val_true_labels_pu = val_true_labels_pu
        self.combined_dataset = combined_dataset
        self.log_dir = log_dir
        self.masked_positive_indices = masked_positive_indices  # Store the original parameter for sklearn's clone
        self._masked_positive_indices_set = set(
            masked_positive_indices) if masked_positive_indices is not None else set()  # Use a private attribute for the set

        # Debug print for masked indices
        if masked_positive_indices is not None:
            print(f"DEBUG: Wrapper initialized with {len(masked_positive_indices)} masked positive indices")
            print(f"DEBUG: Masked indices set size: {len(self._masked_positive_indices_set)}")
        else:
            print("DEBUG: No masked positive indices provided to wrapper")

        self.stc_threshold = stc_threshold
        self.prob_log_path = os.path.join(log_dir, "masked_positives_probabilities_28_07.csv") if self.log_dir else None

        if self.prob_log_path and not os.path.exists(self.prob_log_path):
            with open(self.prob_log_path, 'w') as f:
                f.write("self_training_iteration,sample_index,probability_positive\n")

        self.initial_unlabeled_indices_absolute = initial_unlabeled_indices_absolute

    def _get_or_set_torch_device(self):
        """Helper to determine and return the actual torch.device, setting it if not already set."""
        if self._actual_torch_device_obj is None:
            if self.force_cpu:
                self._actual_torch_device_obj = torch.device("cpu")
            elif self.device_str is not None:
                self._actual_torch_device_obj = torch.device(self.device_str)
            else:
                self._actual_torch_device_obj = torch.device(
                    "cuda" if torch.cuda.is_available() else
                    "mps" if torch.backends.mps.is_available() else
                    "cpu"
                )
            print(f"Resolved actual torch device to: {self._actual_torch_device_obj}")
        return self._actual_torch_device_obj

    def fit(self, X, y, sample_weight=None):
        """
        Model training.
        """
        print(
            f"[DEBUG] Inside ImprovedPyTorchSklearnWrapper.fit (id: {id(self.fit)})\n  X shape: {X.shape}, y shape: {y.shape}\n  Number of positively labeled samples: {(y == 1).sum()}")

        # Track self-training iteration
        if not hasattr(self, '_self_training_iteration'):
            self._self_training_iteration = 0
        else:
            self._self_training_iteration += 1

        # Store reference to self-training estimator if available
        if hasattr(self, '_self_training_estimator'):
            print(f"\nSelf-training iteration {self._self_training_iteration}")
            if hasattr(self._self_training_estimator, 'transduction_'):
                n_unlabeled = np.sum(self._self_training_estimator.transduction_ == -1)
                print(f"Current unlabeled samples: {n_unlabeled}")

        # Determine the actual torch.device to use for training
        actual_device = self._get_or_set_torch_device()
        print(f"Using device for training: {actual_device}")

        # Create model if not provided
        if self.model is None:
            self.model = SimpleSpectraTransformer(
                d_model=self.d_model,
                n_layers=self.n_layers,
                dropout=self.dropout,
                lr=self.lr,
                instrument_embedding_dim=self.instrument_embedding_dim,
                validation_threshold=self.validation_threshold
            )

        # Move the model and all its components to the appropriate device
        self.model = self.model.to(actual_device)

        # Ensure all model components are on the same device
        if hasattr(self.model, 'spectrum_encoder'):
            self.model.spectrum_encoder = self.model.spectrum_encoder.to(actual_device)
            if hasattr(self.model.spectrum_encoder, 'peak_encoder'):
                self.model.spectrum_encoder.peak_encoder = self.model.spectrum_encoder.peak_encoder.to(actual_device)
                if hasattr(self.model.spectrum_encoder.peak_encoder, 'mz_encoder'):
                    self.model.spectrum_encoder.peak_encoder.mz_encoder = (
                        self.model.spectrum_encoder.peak_encoder.mz_encoder.to(actual_device)
                    )

        # Get indices from X
        indices = X.flatten()
        print("the length of indices is :", len(indices))

        # Create a subset of the dataset using indices from X
        if self.dataset is None:
            raise ValueError("Dataset must be provided")

        train_dataset = DynamicLabelDataset(self.dataset, indices, y)

        # Create DataLoader with optimized settings for cluster GPUs
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            persistent_workers=(self.num_workers > 0),
            pin_memory=(self.num_workers > 0)
        )

        # Create validation DataLoader if validation data is provided
        val_loader = None
        metrics_callback = None
        callbacks = []

        if hasattr(self, 'val_indices') and self.val_indices is not None and hasattr(self,
                                                                                     'combined_dataset') and self.combined_dataset is not None:
            print("Setting up validation dataloader...")
            val_dataset_for_loader = Subset(self.combined_dataset, self.val_indices)
            val_loader = DataLoader(
                val_dataset_for_loader,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                collate_fn=collate_fn,
                persistent_workers=(self.num_workers > 0),
                pin_memory=(self.num_workers > 0),
                shuffle=False
            )

            # Add the logging suppression callback first
            callbacks.append(SuppressModelLoggingCallback())

            if hasattr(self, 'log_dir') and self.log_dir is not None and hasattr(self,
                                                                                 'val_true_labels_pu') and self.val_true_labels_pu is not None:
                metrics_callback = SelfTrainingMetricsCallback(
                    wrapper_instance=self,
                    val_indices=self.val_indices,
                    combined_dataset=self.combined_dataset,
                    log_dir=self.log_dir,
                    self_training_iteration=self._self_training_iteration,
                    val_true_labels_pu=self.val_true_labels_pu,
                    initial_unlabeled_indices_absolute=getattr(self, 'initial_unlabeled_indices_absolute', None)
                )
                callbacks.append(metrics_callback)

        # Create PyTorch Lightning trainer with lower log_every_n_steps
        print(f"\n--- Training inner PyTorch model for {self.epochs} epochs ---")

        # accelerator = "cpu" if str(actual_device) == "cpu" else "auto"

        trainer = pl.Trainer(
            max_epochs=self.epochs,
            logger=self.csv_logger,
            accelerator='gpu',
            devices=2,
            strategy=DDPStrategy(gradient_as_bucket_view=True, static_graph=True),
            precision='16-mixed',
            num_sanity_val_steps=0,  # Skip pre-training sanity validation to avoid extra metric rows
            callbacks=callbacks,
            log_every_n_steps=1,  # Log every step
            enable_checkpointing=False,  # Disable checkpointing
            enable_model_summary=True,
            enable_progress_bar=True,
            check_val_every_n_epoch=1  # Validate every epoch
        )

        # Train the model
        if val_loader is not None:
            print("Starting training with validation...")
            trainer.fit(self.model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        else:
            print("Starting training without validation...")
            trainer.fit(self.model, train_dataloaders=train_loader)

        print("--- Inner PyTorch model training completed ---\n")

        self.is_fitted_ = True
        return self

    def predict_proba(self, X, sample_weight=None):
        """
        Predict class probabilities.

        Parameters:
        -----------
        X : array-like, shape = [n_samples, n_features]
            Indices of input data.
        sample_weight : array-like, optional
            Sample weights.

        Returns:
        --------
        array, shape = [n_samples, 2]
            Class probabilities.
        """
        # Check that the model is trained
        check_is_fitted(self, 'is_fitted_')

        # Determine the actual torch.device to use for prediction
        actual_device = self._get_or_set_torch_device()

        # Ensure the model is on the same device as the data
        self.model = self.model.to(actual_device)
        if hasattr(self.model, 'spectrum_encoder'):
            self.model.spectrum_encoder = self.model.spectrum_encoder.to(actual_device)
            if hasattr(self.model.spectrum_encoder, 'peak_encoder'):
                self.model.spectrum_encoder.peak_encoder = self.model.spectrum_encoder.peak_encoder.to(actual_device)
                if hasattr(self.model.spectrum_encoder.peak_encoder, 'mz_encoder'):
                    self.model.spectrum_encoder.peak_encoder.mz_encoder = (
                        self.model.spectrum_encoder.peak_encoder.mz_encoder.to(actual_device)
                    )

        # Set model to evaluation mode
        self.model.eval()

        # Get indices from X
        indices = X.flatten()

        # Create a subset from the dataset using indices from X
        if self.dataset is None:
            raise ValueError("Dataset must be provided")

        test_dataset = Subset(self.dataset, indices)

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            persistent_workers=(self.num_workers > 0),
            pin_memory=(self.num_workers > 0),
            shuffle=False
        )

        all_probs = []
        with torch.no_grad():
            for batch in test_loader:
                # Move data to the appropriate device
                batch = {k: v.to(actual_device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                # Get predictions from the model
                try:
                    # Explicitly move all tensors to the model's device
                    for k, v in batch.items():
                        if isinstance(v, torch.Tensor):
                            batch[k] = v.to(actual_device)

                    # Get predictions
                    logits = self.model(batch)
                    probs = torch.sigmoid(logits)
                    all_probs.append(probs.cpu().numpy())
                except RuntimeError as e:
                    print(f"Error during prediction: {e}")
                    print(f"Model device: {next(self.model.parameters()).device}")
                    for k, v in batch.items():
                        if isinstance(v, torch.Tensor):
                            print(f"Device for {k}: {v.device}")
                    raise

        probs_numpy = np.vstack(all_probs)
        masked_indices_in_this_batch = self._masked_positive_indices_set.intersection(
            indices)

        if masked_indices_in_this_batch:
            # Create a map from the original index to its position in the `probs_numpy` array.
            index_to_prob_map = {original_idx: i for i, original_idx in enumerate(indices)}

            # Prepare data for logging.
            log_rows = []
            newly_labeled_as_pos = 0

            # Get the probabilities for only the masked positive samples.
            for masked_idx in masked_indices_in_this_batch:
                prob_index = index_to_prob_map[masked_idx]
                prob_positive = probs_numpy[prob_index, 0]  # Class 1 prob

                # Check if it would be labeled as positive based on the threshold.
                if prob_positive > self.stc_threshold:
                    newly_labeled_as_pos += 1

                # Append the iteration, index, and score to our log data.
                log_rows.append(f"{self._self_training_iteration},{masked_idx},{prob_positive:.6f}\n")

            newly_labeled_as_neg = len(masked_indices_in_this_batch) - newly_labeled_as_pos

            # Print a summary to the main log.
            summary_msg = (
                f"\n--- [Iteration {self._self_training_iteration} Analysis] ---\n"
                f"    Out of {len(masked_indices_in_this_batch)} masked positives being evaluated:\n"
                f"    - {newly_labeled_as_pos} were predicted as POSITIVE (prob > {self.stc_threshold})\n"
                f"    - {newly_labeled_as_neg} were predicted as NEGATIVE (prob <= {self.stc_threshold})\n"
                f"    - Detailed probabilities saved to {self.prob_log_path}\n"
                f"---------------------------------------\n"
            )
            print(summary_msg)
            logging.info(summary_msg)

            # Write the detailed probability scores to the separate CSV file.
            if self.prob_log_path:
                with open(self.prob_log_path, 'a') as f:
                    f.writelines(log_rows)

        # scikit-learn expects probabilities for both classes: [P(y=0), P(y=1)]
        return np.column_stack([1 - probs_numpy, probs_numpy])

    def predict(self, X, threshold=0.8, sample_weight=None):
        """
        Predict class labels.

        Parameters:
        -----------
        X : array-like, shape = [n_samples, n_features]
            Indices of input data.
        threshold : float, default 0.5
            Decision threshold.
        sample_weight : array-like, optional
            Sample weights.

        Returns:
        --------
        array, shape = [n_samples]
            Predicted class labels.
        """
        probs = self.predict_proba(X)[:, 1]  # Get probabilities for class 1
        return np.array([1.0 if p > threshold else -1.0 for p in probs])

    def set_self_training_estimator(self, estimator):
        """Store reference to the self-training estimator."""
        self._self_training_estimator = estimator

    def validation_step(self, batch, batch_idx):
        """Validation step for the model."""
        logits = self(batch)
        loss = self.bce_loss(logits, batch['labels'])
        probs = torch.sigmoid(logits)

        # Calculate validation metrics
        val_acc = self.val_accuracy(probs, batch['labels'])
        val_prec = self.val_precision(probs, batch['labels'])
        val_recall = self.val_recall(probs, batch['labels'])
        val_f1 = self.val_f1(probs, batch['labels'])

        # Log metrics for this batch
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_accuracy', val_acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_precision', val_prec, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_recall', val_recall, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_f1', val_f1, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        return {'val_loss': loss, 'batch_val_acc': val_acc, 'batch_val_prec': val_prec,
                'batch_val_recall': val_recall, 'batch_val_f1': val_f1}

    def on_validation_epoch_end(self):
        """Called at the end of validation to compute epoch-level metrics."""
        # Get the metrics from the validation step
        val_acc = self.val_accuracy.compute()
        val_prec = self.val_precision.compute()
        val_recall = self.val_recall.compute()
        val_f1 = self.val_f1.compute()

        # Log epoch-level metrics
        self.log('val_accuracy_epoch', val_acc, prog_bar=True)
        self.log('val_precision_epoch', val_prec, prog_bar=True)
        self.log('val_recall_epoch', val_recall, prog_bar=True)
        self.log('val_f1_epoch', val_f1, prog_bar=True)

        # Print metrics
        print(f"\nValidation Epoch End Metrics:")
        print(f"Accuracy: {val_acc:.4f}")
        print(f"Precision: {val_prec:.4f}")
        print(f"Recall: {val_recall:.4f}")
        print(f"F1: {val_f1:.4f}")

        # Reset metrics for next epoch
        self.val_accuracy.reset()
        self.val_precision.reset()
        self.val_recall.reset()
        self.val_f1.reset()


def plot_probability_distribution(probabilities, gmm, epoch, log_dir):
    """
    Plot the probability score distribution with fitted GMM components.
    Fixed version with proper memory cleanup.
    """
    try:
        plt.figure(figsize=(10, 6))

        # Plot histogram of probability scores
        n, bins, patches = plt.hist(probabilities, bins=100, density=True, alpha=0.6, color='gray',
                                    label='Probability Scores')

        # Generate points for GMM components
        x = np.linspace(0, 1, 1000)

        # Plot each GMM component
        means = gmm.means_.flatten()
        covars = gmm.covariances_.flatten()
        weights = gmm.weights_.flatten()

        # Sort components by mean for consistent coloring
        sort_idx = np.argsort(means)
        means = means[sort_idx]
        covars = covars[sort_idx]
        weights = weights[sort_idx]

        # Scale factor to match histogram density
        scale_factor = np.max(n) / np.max([weight * norm.pdf(x, mean, np.sqrt(covar)).max()
                                           for mean, covar, weight in zip(means, covars, weights)])

        colors = ['red', 'blue']
        for i, (mean, covar, weight) in enumerate(zip(means, covars, weights)):
            y = weight * norm.pdf(x, mean, np.sqrt(covar)) * scale_factor
            plt.plot(x, y, color=colors[i],
                     label=f'Component {i + 1} (μ={mean:.3f}, σ={np.sqrt(covar):.3f}, w={weight:.3f})')

        # Plot the full GMM
        y_full = np.zeros_like(x)
        for mean, covar, weight in zip(means, covars, weights):
            y_full += weight * norm.pdf(x, mean, np.sqrt(covar)) * scale_factor
        plt.plot(x, y_full, 'k--', label='Full GMM')

        plt.title(f'Probability Score Distribution (Epoch {epoch})')
        plt.xlabel('Probability Score')
        plt.ylabel('Density (Log Scale)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Set y-axis to log scale and adjust limits for better visibility
        plt.yscale('log')
        min_density_val = np.min(n[n > 0]) * 0.5 if np.any(n > 0) else 1e-4
        plt.ylim(max(min_density_val, 1e-4), np.max(n) * 2.0)

        # Save the plot to a specific subdirectory
        plot_dir = os.path.join(log_dir, 'probability_distributions_28_07_v2')
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(os.path.join(plot_dir, f'probability_distribution_epoch_{epoch}.png'),
                    dpi=300, bbox_inches='tight')

        print(f"Plot saved for epoch {epoch}")

    except Exception as e:
        print(f"Error creating plot for epoch {epoch}: {e}")

    finally:
        # CRITICAL: Always close the figure to free memory
        plt.close('all')  # Close all figures

        # Force garbage collection
        import gc
        gc.collect()


def calculate_pu_metrics(probabilities, true_labels, labeled_pos_indices, log_dir=None, iteration=None, epoch=None):
    print("\n--- calculate_pu_metrics (v2 - Theoretical GMM AUROC) --- DEBUG LOGS ---")
    print(f"Input probabilities (first 10): {probabilities[:10]}")
    print(f"Input true_labels (first 10): {true_labels[:10]}")  # Expected to be 1.0 for P, -1.0 for U
    print(f"Input labeled_pos_indices (first 10): {labeled_pos_indices[:10]}")
    print(
        f"Total samples: {len(probabilities)}, Total true labels: {len(true_labels)}, Total labeled_pos: {len(labeled_pos_indices)}")

    if len(probabilities) != len(true_labels):
        # This check is important, as the error log showed mismatch here.
        print(
            f"ERROR: Length mismatch IN calculate_pu_metrics! probabilities ({len(probabilities)}) != true_labels ({len(true_labels)})")
        raise ValueError(f"Length mismatch: probabilities ({len(probabilities)}) != true_labels ({len(true_labels)})")

    # binary_labels (0/1) are NOT directly used for the theoretical GMM AUROC in this v2 script,
    # but are used by AUPRC/EPR if they were to use roc_auc_score, which they don't directly.
    # The true_labels for this function are expected as 1.0 for P, -1.0 for U by its design for PU metrics.
    # labeled_pos_indices point to the P samples within this true_labels array.

    if not isinstance(labeled_pos_indices, np.ndarray):
        labeled_pos_indices = np.array(labeled_pos_indices)

    # Filter labeled_pos_indices to be within bounds
    valid_labeled_pos_indices = labeled_pos_indices[labeled_pos_indices < len(probabilities)]
    if len(valid_labeled_pos_indices) != len(labeled_pos_indices):
        print(
            f"Warning (v2 script): Some labeled positive indices were out of bounds or invalid. Original: {len(labeled_pos_indices)}, Valid: {len(valid_labeled_pos_indices)}")
    labeled_pos_indices = valid_labeled_pos_indices

    auprc_val = 0.0

    auroc_gmm_val = 0.5  # Default value
    fitted_gmm = None

    print("\n-- GMM AUROC Calculation (Theoretical) --")
    if len(probabilities) < 2:
        print("Warning (v2 script): Not enough samples for GMM fitting (<2). Returning default AUROC GMM.")
    else:
        X = probabilities.reshape(-1, 1)
        try:
            gmm = GaussianMixture(n_components=2, random_state=SEED, reg_covar=1e-6)
            gmm.fit(X)
            fitted_gmm = gmm  # Store the fitted GMM
            means = gmm.means_.flatten()
            variances = gmm.covariances_.flatten()
            stds = np.sqrt(np.maximum(variances, 1e-12))
            weights = gmm.weights_.flatten()
            print(f"GMM Means: {means}, Std Devs: {stds}, Weights: {weights}")
            if means[0] > means[1]:  # Ensure component 0 is "negative" (lower mean)
                means, stds, weights = means[::-1], stds[::-1], weights[::-1]
                print("GMM components swapped for order.")
            mean_neg, std_neg, mean_pos, std_pos = means[0], stds[0], means[1], stds[1]
            print(f"Neg Comp: Mean={mean_neg:.4f}, Std={std_neg:.4f}. Pos Comp: Mean={mean_pos:.4f}, Std={std_pos:.4f}")
            if std_neg < 1e-6 or std_pos < 1e-6:
                print(f"Warning (v2 script): GMM std dev near zero. AUROC GMM might be unreliable. Defaulting to 0.5.")
                auroc_gmm_val = 0.5
            else:
                min_val, max_val = min(0.0, np.min(X) - 0.1), max(1.0, np.max(X) + 0.1)
                thresholds = np.linspace(min_val, max_val, num=500)
                print(
                    f"Thresholds for ROC: min={min_val:.4f}, max={max_val:.4f}, count={len(thresholds)}, sample: {thresholds[:3]}")
                tpr_theoretical = 1 - norm.cdf(thresholds, loc=mean_pos, scale=std_pos)
                fpr_theoretical = 1 - norm.cdf(thresholds, loc=mean_neg, scale=std_neg)
                print(f"Sample TPR_theoretical: {tpr_theoretical[:3]}, Sample FPR_theoretical: {fpr_theoretical[:3]}")

                # Ensure correct sorting for AUC calculation (FPR should be increasing)
                fpr_final = np.concatenate(([0], fpr_theoretical[::-1], [1]))  # Add (0,0) and (1,1) points for AUC
                tpr_final = np.concatenate(([0], tpr_theoretical[::-1], [1]))

                # Remove duplicate FPR values and sort
                unique_fpr, unique_indices = np.unique(fpr_final, return_index=True)
                fpr_final, tpr_final = unique_fpr, tpr_final[unique_indices]

                # Filter out NaN/Inf values
                valid_pts = ~ (np.isnan(fpr_final) | np.isinf(fpr_final) | np.isnan(tpr_final) | np.isinf(tpr_final))
                fpr_final, tpr_final = fpr_final[valid_pts], tpr_final[valid_pts]

                # Ensure points are sorted by FPR for auc function
                sort_order = np.argsort(fpr_final)
                fpr_final, tpr_final = fpr_final[sort_order], tpr_final[sort_order]

                print(f"Final FPR for AUC (first 5): {fpr_final[:5]}, Final TPR for AUC (first 5): {tpr_final[:5]}")
                if len(fpr_final) > 1:
                    auroc_gmm_val = auc(fpr_final, tpr_final)
                else:
                    print(
                        "Warning (v2 script): Not enough valid points for AUC GMM. Defaulting to 0.5.");
                    auroc_gmm_val = 0.5
                print(f"Calculated Theoretical AUROC GMM: {auroc_gmm_val:.4f}")

                # Plot probability distribution if epoch and log_dir are provided
                if epoch is not None and log_dir is not None and fitted_gmm is not None:
                    plot_probability_distribution(probabilities, fitted_gmm, epoch, log_dir)

        except Exception as e:
            print(f"Error during Theoretical GMM AUROC (v2 script): {e}. Defaulting to 0.5.")
            auroc_gmm_val = 0.5

    print("\n-- Percentile Rank & EPR Calculation (v2 script) --")
    if len(labeled_pos_indices) == 0:
        print("Warning (v2 script): No valid labeled positive for AUPR.")
        area_under_percentile_ranks = 0.0
    else:
        if len(probabilities) > 1 and np.max(probabilities) != np.min(
                probabilities):  # Avoid division by zero for uniform probabilities
            ranks = rankdata(probabilities, method='average')
            percentile_ranks = (ranks - 1) / (len(ranks) - 1)
        else:
            percentile_ranks = np.zeros_like(probabilities) if len(probabilities) > 0 else np.array([])
        print(f"Percentile ranks (first 10): {percentile_ranks[:10]}")

        labeled_pos_ranks = percentile_ranks[labeled_pos_indices]
        print(f"Labeled positive ranks (all): {labeled_pos_ranks}")
        if len(labeled_pos_ranks) > 0:
            sorted_ranks = np.sort(labeled_pos_ranks)
            n_pos = len(sorted_ranks)
            print(f"Sorted labeled positive ranks (AUPR): {sorted_ranks}")
            if n_pos > 1:
                x_vals = np.arange(1, n_pos + 1) / n_pos
                area_under_percentile_ranks = np.trapz(sorted_ranks, x=x_vals)
            elif n_pos == 1:
                area_under_percentile_ranks = float(sorted_ranks[0])
            print(f"Calculated Area Under Percentile Ranks (AUPR): {area_under_percentile_ranks:.4f}")
        else:
            print("No labeled_pos_ranks for AUPR.")

    print("--- End calculate_pu_metrics (v2) DEBUG LOGS ---")
    return {'val_auroc_gmm': float(auroc_gmm_val),
            'val_area_under_percentile_ranks': float(area_under_percentile_ranks)}


class SelfTrainingMetricsCallback(pl.Callback):
    def __init__(self, wrapper_instance, val_indices, combined_dataset, log_dir, self_training_iteration,
                 val_true_labels_pu, initial_unlabeled_indices_absolute=None):
        super().__init__()
        self.wrapper_instance = wrapper_instance
        self.val_indices = val_indices
        self.combined_dataset = combined_dataset
        self.log_dir = log_dir
        self.self_training_iteration = self_training_iteration
        self.val_true_labels_pu = val_true_labels_pu
        self.previous_labels = None
        # Save initial unlabeled absolute indices from the wrapper if available
        self.initial_unlabeled_indices_absolute = initial_unlabeled_indices_absolute

    def on_fit_start(self, trainer, pl_module):
        # This method is called at the beginning of each self-training iteration's fit call.
        # We use it to log the label changes from the *previous* iteration.
        if hasattr(self.wrapper_instance, '_self_training_estimator'):
            estimator = self.wrapper_instance._self_training_estimator
            if hasattr(estimator, 'transduction_'):
                current_labels = estimator.transduction_
                iteration = self.wrapper_instance._self_training_iteration

                # Log changes from the previous iteration (if it's not the first one)
                if self.previous_labels is not None and iteration > 0:
                    # Calculate label changes between previous and current state
                    newly_labeled_mask = (self.previous_labels == -1) & (current_labels != -1)
                    total_newly_labeled = np.sum(newly_labeled_mask)

                    # SelfTrainingClassifier predicts classes 0 and 1 from the base estimator
                    newly_labeled_pos = np.sum((self.previous_labels == -1) & (current_labels == 1.0))
                    newly_labeled_neg = np.sum((self.previous_labels == -1) & (current_labels == 0.0))

                    current_unlabeled = np.sum(current_labels == -1)

                    print(f"\n--- Label Changes from Iteration {iteration - 1} ---")
                    print(f"Total newly labeled: {total_newly_labeled}")
                    print(f"  - Newly labeled as positive: {newly_labeled_pos}")
                    print(f"  - Newly labeled as negative: {newly_labeled_neg}")
                    print(f"Remaining unlabeled samples: {current_unlabeled}")

                    # Log metrics for the completed iteration
                    # The global step should correspond to the end of the previous iteration
                    global_step_for_log = (iteration - 1) * self.wrapper_instance.epochs + (
                            self.wrapper_instance.epochs - 1)
                    metrics_to_log = {
                        f"iter_{iteration - 1}/newly_labeled_total": total_newly_labeled,
                        f"iter_{iteration - 1}/newly_labeled_positive": newly_labeled_pos,
                        f"iter_{iteration - 1}/newly_labeled_negative": newly_labeled_neg,
                        f"iter_{iteration - 1}/remaining_unlabeled": current_unlabeled
                    }
                    if trainer.logger:
                        trainer.logger.log_metrics(metrics_to_log, step=global_step_for_log)

                # Store the current labels for the next iteration's comparison
                self.previous_labels = current_labels.copy()

    def on_validation_epoch_end(self, trainer, pl_module):
        current_epoch = trainer.current_epoch
        # Calculate a global step that is unique across all self-training iterations
        global_step = self.self_training_iteration * self.wrapper_instance.epochs + current_epoch
        print(
            f"\n--- Calculating Validation Metrics for Self-Training Iteration {self.self_training_iteration}, Epoch {current_epoch + 1} (Global Step: {global_step}) ---")

        # Get predictions and calculate metrics
        X_val_indices = np.array(self.val_indices).reshape(-1, 1)
        val_probabilities = self.wrapper_instance.predict_proba(X_val_indices)[:, 1]

        val_true_labels_original_format = np.array([self.combined_dataset[i]['label'] for i in self.val_indices])

        # Get predictions on validation set (using 0.5 threshold)
        val_predictions = (val_probabilities > 0.5).astype(int)

        # Calculate counts of predicted positive and negative
        num_predicted_pos = np.sum(val_predictions == 1)
        num_predicted_neg = np.sum(val_predictions == 0)

        # Calculate true positives and false positives among the predicted positives
        true_positives = np.sum((val_predictions == 1) & (val_true_labels_original_format == 1))
        false_positives = np.sum((val_predictions == 1) & (val_true_labels_original_format == 0))

        print("\n--- Validation Set Prediction Stats ---")
        print(f"Predicted as positive: {num_predicted_pos}")
        print(f"  - Actual positives among them (TPs): {true_positives}")
        print(f"  - Actual negatives among them (FPs): {false_positives}")
        print(f"Predicted as negative: {num_predicted_neg}")

        # Add these stats to the metrics to be logged
        validation_stats = {
            'val_predicted_positives': num_predicted_pos,
            'val_predicted_negatives': num_predicted_neg,
            'val_true_positives': true_positives,
            'val_false_positives': false_positives
        }

        val_labeled_pos_indices = np.where(val_true_labels_original_format == 1)[0]

        # ------------------------------------------------------------------
        # Give each probability-distribution plot a unique file name per
        # self-training iteration by embedding the iteration index in the
        # epoch label we pass down.  The label will finally be used by
        # `plot_probability_distribution` when saving the PNG.
        # ------------------------------------------------------------------
        epoch_label = f"iter_{self.self_training_iteration}_epoch_{current_epoch}"

        # Calculate PU metrics and create the plot (if enabled inside the
        # utility function) with the unique label.
        epoch_pu_metrics = calculate_pu_metrics(
            probabilities=val_probabilities,
            true_labels=self.val_true_labels_pu,
            labeled_pos_indices=val_labeled_pos_indices,
            log_dir=self.log_dir,
            iteration=epoch_label,
            epoch=epoch_label,

        )

        # Get standard validation metrics logged by the model during the epoch
        # The SuppressModelLoggingCallback prevents these from being written to the logger directly,
        # but they are available here for us to log with the correct global step.
        val_metrics = {
            'loss': trainer.callback_metrics.get('val_loss_epoch', torch.tensor(0.0)).item(),
            'accuracy': trainer.callback_metrics.get('val_accuracy_epoch', torch.tensor(0.0)).item(),
            'precision': trainer.callback_metrics.get('val_precision_epoch', torch.tensor(0.0)).item(),
            'recall': trainer.callback_metrics.get('val_recall_epoch', torch.tensor(0.0)).item(),
            'f1': trainer.callback_metrics.get('val_f1_epoch', torch.tensor(0.0)).item()
        }

        # Get training metrics logged by the model during the epoch
        train_metrics = {
            'loss': trainer.callback_metrics.get('train_loss_epoch', torch.tensor(0.0)).item(),
            'accuracy': trainer.callback_metrics.get('train_accuracy_epoch', torch.tensor(0.0)).item(),
            'precision': trainer.callback_metrics.get('train_precision_epoch', torch.tensor(0.0)).item(),
            'recall': trainer.callback_metrics.get('train_recall_epoch', torch.tensor(0.0)).item(),
            'f1': trainer.callback_metrics.get('train_f1_epoch', torch.tensor(0.0)).item(),
        }

        # Combine all metrics for logging
        metrics_dict = {}

        # Log validation prediction stats
        for metric_name, value in validation_stats.items():
            metrics_dict[f"iter_{self.self_training_iteration}/val/{metric_name}"] = value

        # Log PU metrics
        for metric_name, value in epoch_pu_metrics.items():
            metrics_dict[f"iter_{self.self_training_iteration}/val/{metric_name}"] = value
            print(f"  {metric_name}: {value:.4f}")

        # Log standard validation metrics
        for metric_name, value in val_metrics.items():
            metrics_dict[f"iter_{self.self_training_iteration}/val/{metric_name}"] = value
            print(f"  val_{metric_name}: {value:.4f}")

        # Log training metrics
        for metric_name, value in train_metrics.items():
            metrics_dict[f"iter_{self.self_training_iteration}/train/{metric_name}"] = value
            print(f"  train_{metric_name}: {value:.4f}")

        # Log all metrics at once with the correct global step
        if trainer.logger:
            trainer.logger.log_metrics(metrics_dict, step=global_step)

        # --- Save probabilities for all initially unlabeled samples at the end of each self-training iteration ---
        if self.initial_unlabeled_indices_absolute is not None and current_epoch == trainer.max_epochs - 1:
            # Get the sample indices (absolute indices into combined_dataset)
            unlabeled_indices = self.initial_unlabeled_indices_absolute
            # Predict probabilities for these samples
            X_unlabeled = unlabeled_indices.reshape(-1, 1)
            unlabeled_probs = self.wrapper_instance.predict_proba(X_unlabeled)[:, 1]
            # Save to CSV
            csv_path = os.path.join(self.log_dir, f"unlabeled_probs_iter_{self.self_training_iteration}.csv")
            with open(csv_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['sample_index', 'probability_positive'])
                for idx, prob in zip(unlabeled_indices, unlabeled_probs):
                    writer.writerow([idx, prob])
            print(f"Saved unlabeled sample probabilities for iteration {self.self_training_iteration} to {csv_path}")

        # --- Save validation probabilities for the last epoch of each self-training iteration ---
        if current_epoch == trainer.max_epochs - 1:
            val_probs_csv_path = os.path.join(self.log_dir, f"val_probs_iter_{self.self_training_iteration}.csv")
            with open(val_probs_csv_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['sample_index', 'probability_positive'])
                for idx, prob in zip(self.val_indices, val_probabilities):
                    writer.writerow([idx, prob])
            print(f"Saved validation probabilities for iteration {self.self_training_iteration} to {val_probs_csv_path}")


def normalize_compound_name(name):
    """
    Normalize compound names by removing collision energy or eV suffixes.
    Handles numeric and None/nan values gracefully.
    """
    # Handle None or nan
    if name is None or (isinstance(name, float) and np.isnan(name)):
        return ""
    # Convert non-string to string (e.g., numbers)
    if not isinstance(name, str):
        name = str(name)
    if "CollisionEnergy:" in name:
        name = name.split("CollisionEnergy:")[0].strip()
    # Remove ' - XX.X eV' or ' - XX eV' at the end
    name = re.sub(r"\s*-\s*\d+(\.\d+)?\s*eV$", "", name.strip())
    return name


def main(args):
    """
    Main function to run the PU learning workflow
    """
    # Initialize logger
    log_dir = os.path.join(args.log_dir, f"pu_learning_experiment_{args.run_name}")
    csv_logger = CSVLogger(log_dir, name="self_training_logs")

    # Load file paths
    with open(args.file_paths, 'r') as f:
        lines = f.readlines()
    mzml_files = [line.split(",")[0].strip() for line in lines]
    csv_files = [line.split(",")[1].strip() for line in lines]

    # Load data from mzML and CSV files
    print("Loading data from mzML and CSV files...")
    datasets = [SimpleMassSpecDataset(mzml, csv) for mzml, csv in zip(mzml_files, csv_files)]
    for i, ds in enumerate(datasets):
        print(f"Dataset {i}: {mzml_files[i]}, {csv_files[i]}, length: {len(ds)}")
    combined_dataset = ConcatDataset(datasets)
    print("Length of dataset:", len(combined_dataset))

    # Get all labels and indices
    all_labels = np.array([combined_dataset.datasets[d_idx].data_pairs[p_idx]['label']
                           for d_idx in range(len(combined_dataset.datasets))
                           for p_idx in range(len(combined_dataset.datasets[d_idx].data_pairs))],
                          dtype=np.int64)

    all_indices = np.arange(len(all_labels))

    # Split data into training and validation sets
    train_indices, val_indices = train_test_split(
        all_indices,
        test_size=args.val_split,
        random_state=args.seed,
        stratify=all_labels
    )

    # Extract original labels for the training set
    train_labels_original = all_labels[train_indices]

    # --- Mask 5% of positives by unique compound name ---
    # Get compound names for all training samples (normalized)
    train_compound_names = np.array([
        normalize_compound_name(combined_dataset.datasets[d_idx].data_pairs[p_idx]['compound_name'])
        for d_idx in range(len(combined_dataset.datasets))
        for p_idx in range(len(combined_dataset.datasets[d_idx].data_pairs))
    ])[train_indices]

    # Find indices of positive samples in the training set
    positive_mask = (train_labels_original == 1)
    positive_indices = np.where(positive_mask)[0]
    positive_compound_names = train_compound_names[positive_indices]

    # Get unique compound names among positives
    unique_compounds = np.unique(positive_compound_names)
    rng = np.random.default_rng(args.seed)
    n_to_mask = max(1, int(0.05 * len(positive_indices)))
    rng.shuffle(unique_compounds)

    # Select compound names to mask (5% of positive samples, by unique compound name)
    masked_compounds = set()
    masked_indices_relative = []  # For pu_labels (relative to training set)
    masked_indices_absolute = []  # For wrapper (absolute dataset indices)
    count = 0
    for cname in unique_compounds:
        # Indices of this compound name among positives (relative to training set)
        relative_idxs = positive_indices[positive_compound_names == cname]
        if count + len(relative_idxs) > n_to_mask:
            continue  # skip if adding this would exceed 5%
        masked_compounds.add(cname)
        # Keep both relative and absolute indices
        masked_indices_relative.extend(relative_idxs)
        absolute_idxs = train_indices[relative_idxs]
        masked_indices_absolute.extend(absolute_idxs)
        count += len(relative_idxs)
        if count >= n_to_mask:
            break
    # Remove masked compound names from the set of unmasked
    unmasked_compounds = set(positive_compound_names) - masked_compounds
    # Ensure no overlap
    assert masked_compounds.isdisjoint(unmasked_compounds)
    # Path for the CSV log
    compound_log_path = os.path.join(args.log_dir, f"masked_unmasked_compounds_{args.run_name}.csv")


    with open(compound_log_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['compound_name', 'status'])  # Header

        for cname in sorted(masked_compounds):
            writer.writerow([cname, 'masked'])
        for cname in sorted(unmasked_compounds):
            writer.writerow([cname, 'unmasked'])

    print(f"Masked/unmasked compound names logged to: {compound_log_path}")

    # --- Create PU labels for the training set: 1 for positive, -1 for unlabeled ---
    pu_labels = np.full_like(train_labels_original, -1, dtype=np.int64)
    pu_labels[positive_mask] = 1
    # Mask the selected positives using relative indices directly
    if masked_indices_relative:
        pu_labels[masked_indices_relative] = -1
        print(
            f"Masked {len(masked_indices_relative)} positive samples as unlabeled, using {len(masked_compounds)} unique compound names.")
        print(f"DEBUG: Total positive samples in training set: {len(positive_indices)}")
        print(f"DEBUG: 5% of positives = {int(0.05 * len(positive_indices))}")
        print(f"DEBUG: Actual masked count = {len(masked_indices_relative)}")
        print(f"DEBUG: Relative masked indices range: {min(masked_indices_relative)} to {max(masked_indices_relative)}")
        print(f"DEBUG: Absolute masked indices range: {min(masked_indices_absolute)} to {max(masked_indices_absolute)}")
    else:
        print("Warning: No positives were masked. Check if there are enough unique compound names.")

    # Update pu_labels using low_probability_indices.txt
    try:
        with open("Scripts/low_probability_indices_2.txt", 'r') as f:
            low_prob_indices = {int(line.strip()) for line in f}

        update_count = 0
        # Iterate over train_indices and update pu_labels in place
        for i, original_idx in enumerate(train_indices):
            if original_idx in low_prob_indices and pu_labels[i] == -1:
                pu_labels[i] = 0
                update_count += 1

        print(f"Updated {update_count} unlabeled samples to label 0 based on low_probability_indices.txt")

    except FileNotFoundError:
        print("Warning: low_probability_indices.txt not found. No labels will be updated.")
    print(f"Number of positive samples in training set: {np.sum(pu_labels == 1.0)}")
    print(f"Number of unlabeled samples in training set: {np.sum(pu_labels == -1.0)}")
    print(f"Number of negative samples in training set: {np.sum(pu_labels == 0)}")
    print(f"Number of samples in training set: {len(pu_labels)}")
    # Get true labels for the validation set, mapped to PU learning context (0 for unlabeled, 1 for positive)
    initial_unlabeled_indices_relative = np.where(pu_labels == -1)[0]
    initial_unlabeled_indices_absolute = train_indices[initial_unlabeled_indices_relative]

    val_true_labels_original = all_labels[val_indices]
    val_true_labels_pu = np.where(val_true_labels_original == 1, 1, 0)

    # Create PyTorch model with optimized parameters for cluster GPUs
    model = SimpleSpectraTransformer(
        d_model=args.d_model,
        n_layers=args.n_layers,
        dropout=args.dropout,
        lr=args.lr,
        instrument_embedding_dim=args.instrument_embedding_dim
    )

    # Create wrapper for PyTorch model with validation data
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
        csv_logger=csv_logger,
        val_indices=val_indices,
        val_true_labels_pu=val_true_labels_pu,
        combined_dataset=combined_dataset,
        log_dir=log_dir,
        validation_threshold=args.validation_threshold,
        masked_positive_indices=masked_indices_absolute,
        stc_threshold=args.threshold,
        initial_unlabeled_indices_absolute=initial_unlabeled_indices_absolute
    )

    # Create SelfTrainingClassifier with our model wrapper
    print("\n--- Initializing SelfTrainingClassifier ---")
    pu_estimator = SelfTrainingClassifier(
        base_estimator=pytorch_model,
        threshold=args.threshold,
        max_iter=args.max_iter,
        criterion='threshold',
        verbose=True
    )

    # Store reference to self-training estimator in the wrapper
    pytorch_model.set_self_training_estimator(pu_estimator)

    print(f"SelfTrainingClassifier: {pu_estimator}")
    print(f"Base classifier: {pytorch_model}")
    print(f"Model device: {next(pytorch_model.model.parameters()).device}")
    print("--- SelfTrainingClassifier initialized ---\n")

    print("--- Starting SelfTrainingClassifier training ---")
    # Convert training indices to an array (each sample index in its own row)
    X_indices = np.array(train_indices).reshape(-1, 1)

    print("Converting data for the SelfTrainingClassifier...")
    print(f"X_indices shape: {X_indices.shape}")
    print(f"pu_labels shape: {pu_labels.shape}")

    # Train the SelfTrainingClassifier using indices and transformed labels
    try:
        pu_estimator.fit(X_indices, pu_labels)
    except Exception as e:
        logging.error(f"Error during SelfTrainingClassifier training: {e}", exc_info=True)
        print(f"Error during SelfTrainingClassifier training: {e}")
        print("Trying to use CPU only for all operations in error recovery...")
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
            csv_logger=csv_logger,
            val_indices=val_indices,
            val_true_labels_pu=val_true_labels_pu,
            combined_dataset=combined_dataset,
            log_dir=log_dir,
            validation_threshold=args.validation_threshold,
            masked_positive_indices=masked_indices_absolute,
            stc_threshold=args.threshold,
            initial_unlabeled_indices_absolute=initial_unlabeled_indices_absolute
        )

        # Create a new PU classifier
        pu_estimator = SelfTrainingClassifier(
            base_estimator=pytorch_model,
            threshold=args.threshold,
            max_iter=args.max_iter,
            criterion='threshold',
            verbose=True
        )

        # Try training again
        pu_estimator.fit(X_indices, pu_labels)

    print("--- SelfTrainingClassifier training completed ---\n")

    # Save the model
    model_save_path = os.path.join(args.log_dir, "self_training_model.pt")
    torch.save(pytorch_model.model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    # --- Post-training Analysis of Labeled Samples ---
    logging.info("\n--- Self-Training Iteration Summary ---")
    labeled_iter = pu_estimator.labeled_iter_
    transduction = pu_estimator.transduction_
    n_iter = pu_estimator.n_iter_

    for i in range(1, n_iter + 1):
        # Find samples that were labeled in the current iteration
        newly_labeled_mask = (labeled_iter == i)

        # Get the labels assigned in this iteration
        new_labels = transduction[newly_labeled_mask]
        new_indices = np.where(newly_labeled_mask)[0]

        # Count new positive (class 1) and negative (class 0) labels
        newly_labeled_pos = np.sum(new_labels == 1.0)
        newly_labeled_neg = np.sum(new_labels == 0.0)

        # --- Masked positives analysis ---
        # masked_indices are indices in the training set, but labeled_iter/transduction are for all training samples
        # So, we need to check which of the newly labeled indices are in masked_indices
        masked_newly_labeled = [idx for idx in new_indices if idx in masked_indices]
        if masked_newly_labeled:
            masked_new_labels = transduction[masked_newly_labeled]
            masked_labeled_pos = np.sum(masked_new_labels == 1.0)
            masked_labeled_neg = np.sum(masked_new_labels == 0.0)
        else:
            masked_labeled_pos = 0
            masked_labeled_neg = 0
        summary_line = f"Iteration {i}: Newly labeled as positive: {newly_labeled_pos}, Newly labeled as negative: {newly_labeled_neg}"
        masked_summary = f"    Of masked positives: labeled as positive: {masked_labeled_pos}, labeled as negative: {masked_labeled_neg}"
        logging.info(summary_line)
        logging.info(masked_summary)
        print(summary_line)
        print(masked_summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_paths", type=str, required=True, help="Path to file containing mzML and CSV paths")
    parser.add_argument("--log_dir", type=str, default="./logs_local_03_07", help="Directory for logs")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for local testing")
    parser.add_argument("--val_split", type=float, default=0.3, help="Validation split")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of DataLoader workers (0 for main process in local testing)")
    parser.add_argument("--d_model", type=int, default=64, help="Model dimension")
    parser.add_argument("--n_layers", type=int, default=2, help="Number of transformer layers for local testing")
    parser.add_argument("--dropout", type=float, default=0.1712215566511, help="Dropout rate")
    parser.add_argument("--lr", type=float, default=0.0004755751039, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=25, help="Number of epochs for inner PyTorch model training")
    parser.add_argument("--instrument_embedding_dim", type=int, default=16,
                        help="Dimension of the instrument embedding output")
    parser.add_argument("--force_cpu", action="store_true", help="Force using CPU instead of GPU/MPS")
    parser.add_argument("--threshold", type=float, default=0.95, help="Threshold for SelfTrainingClassifier")
    parser.add_argument("--max_iter", type=int, default=10, help="Maximum iterations for SelfTrainingClassifier")
    parser.add_argument("--run_name", type=str, default="self_training_29_07_mask_positives_2", help="Name of the run")

    parser.add_argument("--seed", type=int, default=1, help="Random seed for reproducibility")
    parser.add_argument("--validation_threshold", type=float, default=0.95,
                        help="Decision threshold for validation metrics.")
    args = parser.parse_args()
    main(args)