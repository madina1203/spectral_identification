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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, auc
from sklearn.mixture import GaussianMixture
from scipy.stats import rankdata, norm # Added norm
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

        if force_cpu:
            self.device = torch.device("cpu")
        elif device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else
                                       "mps" if torch.backends.mps.is_available() else
                                       "cpu")
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")
        self.model = self.model.to(self.device)

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
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                self.model = self.model.to(self.device)
                logits = self.model(batch)
                probs = torch.sigmoid(logits)
                all_probs.append(probs.cpu().numpy())
        probs_numpy = np.vstack(all_probs)
        return np.column_stack([1 - probs_numpy, probs_numpy])

    def fit(self, train_indices, val_indices=None, callbacks=None):
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
            precision='16-mixed'
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
        self.val_batch_indices = []

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if isinstance(outputs, dict) and 'logits' in outputs:
            self.val_outputs.append(outputs['logits'])
            start_idx = batch_idx * trainer.val_dataloaders.batch_size
            end_idx = min(start_idx + trainer.val_dataloaders.batch_size, len(self.val_indices))
            self.val_batch_indices.extend(range(start_idx, end_idx))
        else:
            print("Warning: Validation step did not return logits in expected format")

    def on_validation_epoch_end(self, trainer, pl_module):
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

        if self.val_outputs:
            val_probs_s1 = torch.cat(self.val_outputs).sigmoid().cpu().numpy()
            val_true_labels = np.array([self.dataset[self.val_indices[i]]['label'] for i in self.val_batch_indices])
        else:
            print("Warning: No validation outputs collected, falling back to wrapper prediction")
            val_probs_s1 = self.wrapper.predict_proba(self.val_indices)[:, 1]
            val_true_labels = np.array([self.dataset[i]['label'] for i in self.val_indices])

        val_probs_y1_adjusted = np.clip(val_probs_s1 / estimated_c, 0, 1)
        val_predictions_pu_adjusted = np.array([1.0 if p > 0.5 else -1.0 for p in val_probs_y1_adjusted])
        val_true_labels_pu = np.array([1.0 if label == 1 else -1.0 for label in val_true_labels])

        val_labeled_pos_indices = np.where(val_true_labels == 1)[0]
        additional_pu_metrics = calculate_pu_metrics(
            val_probs_y1_adjusted, # Use adjusted probabilities for PU metrics
            val_true_labels_pu,
            val_labeled_pos_indices
        )

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

        trainer.logger.log_metrics(metrics, step=trainer.current_epoch)
        print(f"\\nEpoch {trainer.current_epoch} PU Metrics (Fixed Holdout):")
        for name, value in metrics.items():
            print(f"  {name}: {value:.4f}")

        self.val_outputs = []
        self.val_batch_indices = []


def calculate_pu_metrics(probabilities, true_labels, labeled_pos_indices):
    print("\n--- calculate_pu_metrics (v2 - Theoretical GMM AUROC) --- DEBUG LOGS ---")
    print(f"Input probabilities (first 10): {probabilities[:10]}")
    print(f"Input true_labels (first 10): {true_labels[:10]}")
    print(f"Input labeled_pos_indices (first 10): {labeled_pos_indices[:10]}")
    print(f"Total samples: {len(probabilities)}, Total true labels: {len(true_labels)}, Total labeled_pos: {len(labeled_pos_indices)}")

    if len(probabilities) != len(true_labels):
        raise ValueError(f"Length mismatch: probabilities ({len(probabilities)}) != true_labels ({len(true_labels)})")

    binary_labels = np.array([1 if label == 1 else 0 for label in true_labels]) # Not used for theoretical GMM AUROC, but kept for other metrics if any
    print(f"Binary labels (first 10, for potential other metrics): {binary_labels[:10]}")

    if not isinstance(labeled_pos_indices, np.ndarray):
        labeled_pos_indices = np.array(labeled_pos_indices)

    if len(labeled_pos_indices) > 0 and np.any(labeled_pos_indices >= len(probabilities)):
        print(f"Warning: Some labeled positive indices are out of bounds. Adjusting indices...")
        original_count = len(labeled_pos_indices)
        labeled_pos_indices = labeled_pos_indices[labeled_pos_indices < len(probabilities)]
        print(f"Adjusted labeled_pos_indices count from {original_count} to {len(labeled_pos_indices)}")

    auprc_val = 0.0
    epr_val = 0.0
    auroc_gmm_val = 0.5 # Default value

    # GMM-based AUROC (theoretical calculation based on R code)
    print("\n-- GMM AUROC Calculation (Theoretical) --")
    if len(probabilities) < 2:
        print("Warning: Not enough samples for GMM fitting (<2). Returning default AUROC GMM.")
    else:
        X = probabilities.reshape(-1, 1)
        try:
            gmm = GaussianMixture(n_components=2, random_state=SEED, reg_covar=1e-6)
            gmm.fit(X)

            means = gmm.means_.flatten()
            variances = gmm.covariances_.flatten()
            stds = np.sqrt(np.maximum(variances, 1e-12)) # Ensure stds are non-negative and non-zero
            weights = gmm.weights_.flatten()
            print(f"GMM Means: {means}")
            print(f"GMM Std Devs (sqrt of max(variances, 1e-12)): {stds}")
            print(f"GMM Weights: {weights}")

            # Ensure component 0 is "negative" (lower mean) and component 1 is "positive" (higher mean)
            if means[0] > means[1]:
                means = means[::-1]
                stds = stds[::-1]
                weights = weights[::-1]
                print("GMM components swapped to ensure mean[0] < mean[1]")
            
            mean_neg, std_neg = means[0], stds[0]
            mean_pos, std_pos = means[1], stds[1]
            print(f"Negative Component: Mean={mean_neg:.4f}, Std={std_neg:.4f}")
            print(f"Positive Component: Mean={mean_pos:.4f}, Std={std_pos:.4f}")

            if std_neg < 1e-6 or std_pos < 1e-6:
                 print(f"Warning: GMM component standard deviation is near zero (neg_std:{std_neg:.4g}, pos_std:{std_pos:.4g}). Theoretical AUROC GMM might be unreliable. Defaulting to 0.5.")
                 auroc_gmm_val = 0.5
            else:
                min_val = min(0.0, np.min(X) - 0.1)
                max_val = max(1.0, np.max(X) + 0.1)
                thresholds = np.linspace(min_val, max_val, num=500)
                print(f"Thresholds for theoretical ROC: min={min_val:.4f}, max={max_val:.4f}, count={len(thresholds)}")
                print(f"Sample thresholds (first 5): {thresholds[:5]}")

                tpr_theoretical = 1 - norm.cdf(thresholds, loc=mean_pos, scale=std_pos)
                fpr_theoretical = 1 - norm.cdf(thresholds, loc=mean_neg, scale=std_neg)
                print(f"Theoretical TPR (first 5): {tpr_theoretical[:5]}")
                print(f"Theoretical FPR (first 5): {fpr_theoretical[:5]}")

                fpr_sorted = fpr_theoretical[::-1]
                tpr_sorted = tpr_theoretical[::-1]
                
                fpr_final = np.concatenate(([0], fpr_sorted, [1]))
                tpr_final = np.concatenate(([0], tpr_sorted, [1]))

                unique_fpr, unique_indices = np.unique(fpr_final, return_index=True)
                fpr_final = unique_fpr
                tpr_final = tpr_final[unique_indices]
                print(f"FPR for AUC (first 10 after unique and sort): {fpr_final[:10]}")
                print(f"TPR for AUC (first 10 after unique and sort): {tpr_final[:10]}")
                
                valid_pts = ~ (np.isnan(fpr_final) | np.isinf(fpr_final) | np.isnan(tpr_final) | np.isinf(tpr_final))
                fpr_final = fpr_final[valid_pts]
                tpr_final = tpr_final[valid_pts]

                sort_order = np.argsort(fpr_final)
                fpr_final = fpr_final[sort_order]
                tpr_final = tpr_final[sort_order]

                if len(fpr_final) > 1 and len(tpr_final) > 1:
                    auroc_gmm_val = auc(fpr_final, tpr_final)
                    print(f"Calculated Theoretical AUROC GMM: {auroc_gmm_val:.4f}")
                else:
                    print("Warning: Not enough valid points to calculate Theoretical AUROC GMM. Defaulting to 0.5.")
                    auroc_gmm_val = 0.5
        
        except ValueError as e:
            print(f"ValueError during GMM fitting or Theoretical AUROC calculation: {e}. Defaulting AUROC GMM to 0.5.")
            auroc_gmm_val = 0.5
        except Exception as e:
            print(f"Unexpected error during Theoretical GMM AUROC calculation: {e}. Defaulting AUROC GMM to 0.5.")
            auroc_gmm_val = 0.5

    # Percentile rank and EPR calculation (remains the same)
    print("\n-- Percentile Rank & EPR Calculation --")
    if len(labeled_pos_indices) == 0:
        print("Warning: No valid labeled positive indices for AUPRC/EPR. Returning default values for AUPRC/EPR.")
    else:
        if len(probabilities) > 1 and np.max(probabilities) != np.min(probabilities):
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
            print(f"Sorted labeled positive ranks (for AUPRC): {sorted_ranks}")
            if n_pos > 1:
                x_values_for_auprc = np.arange(1, n_pos + 1) / n_pos
                print(f"X-values for AUPRC: {x_values_for_auprc}")
                auprc_val = np.trapz(sorted_ranks, x=x_values_for_auprc)
            elif n_pos == 1:
                 auprc_val = float(sorted_ranks[0])
            print(f"Calculated AUPRC: {auprc_val:.4f}")
        else:
            print("No labeled positive ranks to calculate AUPRC.")
            auprc_val = 0.0 # Ensure it's explicitly 0 if no labeled_pos_ranks

        k = 0.1
        threshold_epr = 1 - k
        print(f"EPR k: {k}, EPR threshold (percentile > {threshold_epr})")
        if len(labeled_pos_ranks) > 0:
            epr_val = np.mean(labeled_pos_ranks > threshold_epr)
            print(f"EPR (mean of labeled_pos_ranks > {threshold_epr}): {epr_val:.4f}")
        else:
            print("No labeled positive ranks to calculate EPR.")
            epr_val = 0.0 # Ensure it's explicitly 0

    print("--- End calculate_pu_metrics (v2) DEBUG LOGS ---")
    return {
        'auroc_gmm': float(auroc_gmm_val),
        'auprc': float(auprc_val),
        'epr': float(epr_val)
    }


def main(args):
    if torch.cuda.is_available():
        print(f"Available GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    experiment_name = "pu_learning_gmm_v2_train_val_only_23_05" # Modified experiment name
    log_dir = os.path.join(args.log_dir, experiment_name)
    os.makedirs(log_dir, exist_ok=True)
    csv_logger = CSVLogger(
        save_dir=args.log_dir,
        name=experiment_name,
        version=datetime.datetime.now().strftime("%Y%m%d_%H%M%S"), # Unique version
        flush_logs_every_n_steps=10
    )
    with open(args.file_paths, 'r') as f:
        lines = f.readlines()
    mzml_files = [line.split(",")[0].strip() for line in lines]
    csv_files = [line.split(",")[1].strip() for line in lines]
    datasets = [SimpleMassSpecDataset(mzml, csv) for mzml, csv in zip(mzml_files, csv_files)]
    combined_dataset = ConcatDataset(datasets)
    labels = [combined_dataset[i]['label'] for i in range(len(combined_dataset))]
    labels = np.array(labels, dtype=np.int64)
    indices = np.arange(len(combined_dataset))

    print(f"Total samples: {len(indices)}")
    print(f"Positive samples: {np.sum(labels == 1)}")
    print(f"Unlabeled samples: {np.sum(labels == 0)}")

    if np.sum(labels == 1) == 0:
        raise ValueError("No positive samples found in the dataset!")

    # Split: train/val using stratified split (70% train, 30% val)
    train_indices, val_indices = train_test_split(
        indices, # Use all indices
        test_size=0.3, # 30% for validation
        random_state=SEED,
        stratify=labels # Stratify on all labels
    )

    train_labels = labels[train_indices]
    val_labels = labels[val_indices]

    print("\\nSplit Statistics:")
    print(f"Training set: {len(train_indices)} samples, {np.sum(train_labels == 1)} positives")
    print(f"Validation set: {len(val_indices)} samples, {np.sum(val_labels == 1)} positives")

    # --- Fixed hold-out logic (applied to the 70% training set) ---
    positive_train_indices = train_indices[train_labels == 1]
    hold_out_ratio = args.hold_out_ratio

    if len(positive_train_indices) == 0:
        print("Warning: No positive samples in the initial training split to create a hold-out set.")
        holdout_pos_indices = np.array([], dtype=int)
        train_indices_no_holdout = train_indices.copy()
    else:
        n_hold_out = max(1, int(np.ceil(len(positive_train_indices) * hold_out_ratio)))
        if n_hold_out >= len(positive_train_indices):
            n_hold_out = len(positive_train_indices) - 1 if len(positive_train_indices) > 1 else 0
            if n_hold_out < 1 and len(positive_train_indices) > 1:
                 print("Warning: Adjusted n_hold_out to be less than total positives in training. Hold-out set might be very small or empty.")

        if n_hold_out == 0:
            print("Warning: Calculated n_hold_out is 0. No samples for hold-out set from training positives. C-estimation might be impacted.")
            holdout_pos_indices = np.array([], dtype=int)
            train_indices_no_holdout = train_indices.copy()
        else:
            if len(positive_train_indices) > n_hold_out:
                 _, holdout_pos_indices = train_test_split(
                    positive_train_indices,
                    test_size=n_hold_out,
                    random_state=SEED
                )
                 train_indices_no_holdout = np.setdiff1d(train_indices, holdout_pos_indices)
            elif len(positive_train_indices) > 0:
                 print(f"Warning: Not enough positive samples ({len(positive_train_indices)}) for the desired hold_out_ratio resulting in n_hold_out={n_hold_out}. Using all available positives for holdout.")
                 holdout_pos_indices = positive_train_indices.copy()
                 train_indices_no_holdout = np.setdiff1d(train_indices, holdout_pos_indices)
            else:
                 holdout_pos_indices = np.array([], dtype=int)
                 train_indices_no_holdout = train_indices.copy()
    
    print(f"\\nHold-out set: {len(holdout_pos_indices)} positive samples (from training set)")
    print(f"Training set after hold-out: {len(train_indices_no_holdout)} samples")

    train_labels_no_holdout = labels[train_indices_no_holdout]
    if np.sum(train_labels_no_holdout == 1) == 0 and len(train_indices_no_holdout) > 0:
        print("Warning: No positive samples left in training set after hold-out!")

    model = SimpleSpectraTransformer(
        d_model=args.d_model, n_layers=args.n_layers, dropout=args.dropout,
        lr=args.lr, instrument_embedding_dim=args.instrument_embedding_dim
    )
    
    wrapper_args = dict(
        model=model, dataset=combined_dataset, d_model=args.d_model, n_layers=args.n_layers,
        dropout=args.dropout, lr=args.lr, batch_size=args.batch_size, epochs=args.epochs,
        num_workers=args.num_workers, instrument_embedding_dim=args.instrument_embedding_dim,
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

    pu_callback = FixedHoldoutPUCallback(
        wrapper=pytorch_model, train_indices=train_indices_no_holdout,
        val_indices=val_indices, holdout_pos_indices=holdout_pos_indices, dataset=combined_dataset
    )

    pytorch_model.fit(
        train_indices=train_indices_no_holdout, val_indices=val_indices, callbacks=[pu_callback]
    )
    
    print("\\nTraining finished.")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_path = os.path.join(log_dir, f"pu_model_v2_{timestamp}.pt") # Keep v2 in model name
    torch.save(pytorch_model.model.state_dict(), model_save_path)
    
    summary_path = os.path.join(log_dir, f"experiment_summary_v2_{timestamp}.txt") # Keep v2 in summary name
    with open(summary_path, 'w') as f:
        f.write("=== Experiment Summary (v2 GMM AUROC, Train/Val Only) ===\\n\\n")
        f.write("Hyperparameters:\\n")
        for param, value in vars(args).items():
            f.write(f"{param}: {value}\\n")
        # Validation metrics are logged by the callback and CSVLogger

    print(f"\\nModel saved to {model_save_path}")
    print(f"Experiment summary saved to {summary_path}")
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
    parser.add_argument("--instrument_embedding_dim", type=int, default=16, help="Dimension of the instrument embedding output")
    parser.add_argument("--force_cpu", action="store_true", help="Force using CPU instead of GPU/MPS")
    args = parser.parse_args()
    main(args) 