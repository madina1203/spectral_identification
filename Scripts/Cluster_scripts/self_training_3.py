import os
import datetime
import sys
import time

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import torch.nn as nn
import lightning.pytorch as pl
from torch.utils.data import DataLoader, Dataset, Subset, ConcatDataset
import numpy as np
import argparse
from lightning.pytorch.loggers import CSVLogger
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, auc
from sklearn.semi_supervised import SelfTrainingClassifier
from lightning.pytorch.callbacks import Callback
from sklearn.mixture import GaussianMixture
from scipy.stats import norm, rankdata
from lightning.pytorch.strategies import DDPStrategy
import matplotlib.pyplot as plt

# --- Ensure src path is included ---
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.transformers.simpleDataset import SimpleMassSpecDataset
from src.transformers.model_simplified import SimpleSpectraTransformer

# Set a seed for reproducibility
SEED = 1
pl.seed_everything(SEED, workers=True)


# The collate_fn remains the same
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


# Helper class to dynamically change labels for self-training
class LabelOverriddenDataset(Dataset):
    def __init__(self, original_dataset, new_labels):
        self.original_dataset = original_dataset
        self.new_labels = new_labels
        assert len(self.original_dataset) == len(self.new_labels)

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        item = self.original_dataset[idx]
        item['label'] = self.new_labels[idx]
        return item


class SelfTrainingPyTorchWrapper:
    def __init__(self, model_class, model_params, dataset, batch_size=64, epochs=5,
                 num_workers=4, accelerator="gpu", devices=2):
        self.model_class = model_class
        self.model_params = model_params
        self.model = self._create_new_model()
        self.dataset = dataset
        self.batch_size = batch_size
        self.epochs = epochs
        self.num_workers = num_workers
        self.accelerator = accelerator
        self.devices = devices

        # The device will be managed by PyTorch Lightning's trainer.
        # self.model is not moved to a device here.
        self.is_fitted_ = False
        self.classes_ = np.array([0, 1])

    def get_params(self, deep=True):
        return {
            'model_class': self.model_class,
            'model_params': self.model_params,
            'dataset': self.dataset,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'num_workers': self.num_workers,
            'accelerator': self.accelerator,
            'devices': self.devices,
        }

    def _create_new_model(self):
        """Creates a fresh, untrained model instance."""
        return self.model_class(**self.model_params)

    def predict_proba(self, indices):
        # sklearn passes X as a 2D array, we ravel it to get our 1D indices
        if isinstance(indices, np.ndarray) and indices.ndim == 2:
            indices = indices.ravel()

        self.model.eval()
        # For prediction, we can use a single GPU or CPU.
        if torch.cuda.is_available() and self.accelerator == 'gpu':
            device = torch.device(f"cuda:0")
        else:
            device = torch.device("cpu")
        self.model = self.model.to(device)

        test_dataset = Subset(self.dataset, indices)
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )
        all_probs = []
        with torch.no_grad():
            for batch in test_loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                logits = self.model(batch)
                probs = torch.sigmoid(logits)
                all_probs.append(probs.cpu().numpy())
        probs_numpy = np.vstack(all_probs)
        return np.column_stack([1 - probs_numpy, probs_numpy])

    def fit(self, train_indices, train_labels, val_indices=None, logger=None):
        if isinstance(train_indices, np.ndarray) and train_indices.ndim == 2:
            train_indices = train_indices.ravel()

        print(f"Fitting on {len(train_indices)} samples.")

        self.model = self._create_new_model()  # Don't move to device manually

        train_subset = Subset(self.dataset, train_indices)
        # Use provided labels, which might be pseudo-labels
        training_dataset_with_custom_labels = LabelOverriddenDataset(train_subset, train_labels)

        train_loader = DataLoader(
            training_dataset_with_custom_labels,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            shuffle=True,
            persistent_workers=True if self.num_workers > 0 else False,
            pin_memory=True
        )

        callbacks = []
        val_loader = None
        if val_indices is not None and logger is not None:
            val_subset = Subset(self.dataset, val_indices)
            val_loader = DataLoader(val_subset, batch_size=self.batch_size, num_workers=self.num_workers,
                                    collate_fn=collate_fn, persistent_workers=True if self.num_workers > 0 else False,
                                    pin_memory=True)
            pu_callback = PUMetricsCallback()
            callbacks.append(pu_callback)

        strategy = DDPStrategy(gradient_as_bucket_view=True, static_graph=True) if self.devices > 1 else 'auto'

        trainer = pl.Trainer(
            max_epochs=self.epochs,
            accelerator=self.accelerator,
            devices=self.devices,
            strategy=strategy,
            precision='16-mixed',
            enable_progress_bar=True,
            logger=logger,
            callbacks=callbacks,
            log_every_n_steps=max(1, len(train_loader) // 2)  # Avoid too many logs
        )

        trainer.fit(self.model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        self.is_fitted_ = True
        return self


class PUMetricsCallback(Callback):
    def __init__(self):
        super().__init__()
        self.validation_step_outputs = []

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if isinstance(outputs, dict) and 'logits' in outputs and 'labels' in outputs:
            self.validation_step_outputs.append({
                'logits': outputs['logits'].detach().cpu(),
                'labels': outputs['labels'].detach().cpu()
            })

    def on_validation_epoch_end(self, trainer, pl_module):
        if not self.validation_step_outputs:
            return

        all_logits = torch.cat([out['logits'] for out in self.validation_step_outputs])
        all_labels = torch.cat([out['labels'] for out in self.validation_step_outputs]).numpy()

        val_probs_s1 = all_logits.sigmoid().cpu().numpy()

        # --- New: Calculate all requested PU val metrics ---
        val_preds = (val_probs_s1 > 0.5).astype(int)
        pu_val_accuracy = accuracy_score(all_labels, val_preds)

        # Prepare labels for other PU metrics: P=1, U=-1
        val_labels_for_pu = np.where(all_labels == 1, 1.0, -1.0)
        val_pos_indices_in_val_set = np.where(all_labels == 1)[0]

        # Determine log directory for plots if available
        log_dir = trainer.logger.log_dir if trainer.logger and hasattr(trainer.logger, "log_dir") else None

        other_pu_metrics = calculate_pu_metrics(
            probabilities=val_probs_s1,
            true_labels=val_labels_for_pu,
            labeled_pos_indices=val_pos_indices_in_val_set,
            epoch=trainer.current_epoch,
            log_dir=log_dir
        )

        metrics_to_log = {
            'pu_val_accuracy': pu_val_accuracy,
            'pu_val_auprc': other_pu_metrics['auprc'],
            'pu_val_auroc_gmm': other_pu_metrics['auroc_gmm'],
            'pu_val_epr': other_pu_metrics['epr'],
        }

        if trainer.logger:
            trainer.logger.log_metrics(metrics_to_log)  # The logger adds epoch/step automatically

        print(f"\n--- Epoch {trainer.current_epoch} PU Metrics ---")
        for name, value in metrics_to_log.items():
            print(f"  {name}: {value:.4f}")

        self.validation_step_outputs.clear()


# --------------------
# Helper: Plot probability distribution with fitted GMM components
# --------------------

def plot_probability_distribution(probabilities, gmm, epoch, log_dir):
    """Plot histogram of probability scores together with each fitted GMM component.
    The figure is stored as PNG:  {log_dir}/probability_distribution_epoch_{epoch}.png
    The implementation is a verbatim copy of the helper used in
    `train_pulearn_gmm_distribution_check_2.py`.  DO NOT EDIT without syncing both files.
    """
    try:
        plt.figure(figsize=(10, 6))
        # Histogram (density=True ⇒ area=1 so it aligns with pdf)
        n, bins, _ = plt.hist(probabilities, bins=100, density=True, alpha=0.6,
                              color='gray', label='Probability Scores')

        x = np.linspace(0, 1, 1000)
        means = gmm.means_.flatten()
        covars = gmm.covariances_.flatten()
        weights = gmm.weights_.flatten()
        # Sort for deterministic colouring
        order = np.argsort(means)
        means, covars, weights = means[order], covars[order], weights[order]
        # Scale factor so individual pdfs roughly match histogram peak
        scale = np.max(n) / np.max([w * norm.pdf(x, m, np.sqrt(c)).max()
                                    for m, c, w in zip(means, covars, weights)])
        colours = ['red', 'blue']
        for i, (m, c, w) in enumerate(zip(means, covars, weights)):
            y = w * norm.pdf(x, m, np.sqrt(c)) * scale
            plt.plot(x, y, color=colours[i % len(colours)],
                     label=f'Component {i+1} (μ={m:.3f}, σ={np.sqrt(c):.3f}, w={w:.3f})')
        # Full mixture
        y_full = np.zeros_like(x)
        for m, c, w in zip(means, covars, weights):
            y_full += w * norm.pdf(x, m, np.sqrt(c)) * scale
        plt.plot(x, y_full, 'k--', label='Full GMM')

        plt.title(f'Probability Score Distribution (Epoch {epoch})')
        plt.xlabel('Probability Score')
        plt.ylabel('Density (log scale)')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Create directory & save
        os.makedirs(log_dir, exist_ok=True)
        filepath = os.path.join(log_dir, f'probability_distribution_epoch_{epoch}.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved probability-distribution plot ➜ {filepath}")
    except Exception as e:
        print(f"Error while plotting probability distribution (epoch {epoch}): {e}")
    finally:
        plt.close('all')
        import gc; gc.collect()


def calculate_pu_metrics(probabilities, true_labels, labeled_pos_indices, epoch=None, log_dir=None):
    print("\n--- calculate_pu_metrics ---")

    auprc_val = 0.0
    epr_val = 0.0
    auroc_gmm_val = 0.5  # Default value

    print("\n-- GMM AUROC Calculation (Theoretical) --")
    fitted_gmm = None
    if len(probabilities) < 2:
        print("Warning: Not enough samples for GMM fitting (<2). Returning default AUROC GMM.")
    else:
        X = probabilities.reshape(-1, 1)
        try:
            gmm = GaussianMixture(n_components=2, random_state=SEED, reg_covar=1e-6)
            gmm.fit(X)
            fitted_gmm = gmm  # keep for plotting
            means = gmm.means_.flatten()
            variances = gmm.covariances_.flatten()
            stds = np.sqrt(np.maximum(variances, 1e-12))
            weights = gmm.weights_.flatten()
            if means[0] > means[1]:  # Ensure component-0 is negative
                means, stds, weights = means[::-1], stds[::-1], weights[::-1]
            mean_neg, std_neg, mean_pos, std_pos = means[0], stds[0], means[1], stds[1]
            if std_neg < 1e-6 or std_pos < 1e-6:
                auroc_gmm_val = 0.5
            else:
                thresholds = np.linspace(min(0.0, probabilities.min()-0.1),
                                         max(1.0, probabilities.max()+0.1), 500)
                tpr = 1 - norm.cdf(thresholds, loc=mean_pos, scale=std_pos)
                fpr = 1 - norm.cdf(thresholds, loc=mean_neg, scale=std_neg)
                fpr_final = np.concatenate(([0], fpr[::-1], [1]))
                tpr_final = np.concatenate(([0], tpr[::-1], [1]))
                uniq_fpr, idx = np.unique(fpr_final, return_index=True)
                fpr_final, tpr_final = uniq_fpr, tpr_final[idx]
                mask = ~(np.isnan(fpr_final) | np.isinf(fpr_final) | np.isnan(tpr_final) | np.isinf(tpr_final))
                fpr_final, tpr_final = fpr_final[mask], tpr_final[mask]
                sort_idx = np.argsort(fpr_final)
                fpr_final, tpr_final = fpr_final[sort_idx], tpr_final[sort_idx]
                if len(fpr_final) > 1:
                    auroc_gmm_val = auc(fpr_final, tpr_final)
        except Exception as e:
            print(f"Error during GMM AUROC computation: {e}. Defaulting to 0.5.")
            auroc_gmm_val = 0.5

    # Optional plotting
    if epoch is not None and log_dir is not None and fitted_gmm is not None:
        plot_probability_distribution(probabilities, fitted_gmm, epoch, log_dir)

    print("\n-- Percentile Rank & EPR Calculation --")
    if len(labeled_pos_indices) > 0:
        ranks = rankdata(probabilities, method='average') if probabilities.max() != probabilities.min() else np.zeros_like(probabilities)
        percentile_ranks = (ranks - 1) / (len(ranks) - 1) if len(ranks) > 1 else np.zeros_like(probabilities)
        labeled_pos_ranks = percentile_ranks[labeled_pos_indices]
        if len(labeled_pos_ranks) > 0:
            sorted_ranks = np.sort(labeled_pos_ranks)
            n_pos = len(sorted_ranks)
            if n_pos > 1:
                x_vals = np.arange(1, n_pos + 1) / n_pos
                auprc_val = np.trapz(sorted_ranks, x=x_vals)
            else:
                auprc_val = float(sorted_ranks[0])
            epr_val = np.mean(labeled_pos_ranks > 0.9)  # top-10-percent cutoff

    return {'auroc_gmm': float(auroc_gmm_val), 'auprc': float(auprc_val), 'epr': float(epr_val)}


def main(args):
    # --- Logger ---
    log_dir_base = args.log_dir
    log_name = f"self_training_v2_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    csv_logger = CSVLogger(log_dir_base, name=log_name)

    # --- Dataset Loading ---
    print("Loading datasets...")
    # mzml_files = [
    #     os.path.join(args.data_dir, "SP_Orbitrap_Velos_Pro_DI_2_Negative_20_RAW_20.mzML"),
    #     os.path.join(args.data_dir, "SP_Orbitrap_Velos_Pro_DI_2_Positive_20_RAW_20.mzML")
    # ]
    # csv_files = [
    #     os.path.join(args.data_dir, "SP_Orbitrap_Velos_Pro_DI_2_Negative_20_RAW_20.csv"),
    #     os.path.join(args.data_dir, "SP_Orbitrap_Velos_Pro_DI_2_Positive_20_RAW_20.csv")
    # ]
    with open(args.file_paths, 'r') as f:
        lines = f.readlines()
    mzml_files = [line.split(",")[0].strip() for line in lines]
    csv_files = [line.split(",")[1].strip() for line in lines]

    datasets = [SimpleMassSpecDataset(mzml, csv) for mzml, csv in zip(mzml_files, csv_files)]
    combined_dataset = ConcatDataset(datasets)
    print(f"Total dataset size: {len(combined_dataset)}")

    # --- Data Splitting ---
    all_indices = np.arange(len(combined_dataset))
    original_labels = np.array([combined_dataset[i]['label'] for i in all_indices])
    pos_indices = all_indices[original_labels == 1]
    unlabeled_indices = all_indices[original_labels == 0]

    # Split positive samples into a small labeled set and a set to be part of the unlabeled pool
    labeled_pos_indices, unlabeled_pos_indices = train_test_split(
        pos_indices,
        test_size=args.unlabeled_pos_proportion,
        random_state=args.seed
    )

    # Combine the majority of positives (now unlabeled) with the original negatives
    unlabeled_pool_indices = np.concatenate([unlabeled_indices, unlabeled_pos_indices])

    # Create the initial training set and a final hold-out test set from the unlabeled pool
    # The 'test set' for sklearn's fit method is our validation set.
    train_unlabeled_indices, val_indices = train_test_split(
        unlabeled_pool_indices,
        test_size=args.val_split,
        random_state=args.seed
    )

    # The initial labeled set for the very first fit call.
    initial_labeled_indices = labeled_pos_indices
    y_initial_labeled = np.ones(len(initial_labeled_indices), dtype=int)

    # The unlabeled dataset for self-training. The model will predict labels for these.
    X_unlabeled = train_unlabeled_indices.reshape(-1, 1)

    print(f"Initial labeled positive size: {len(initial_labeled_indices)}")
    print(f"Unlabeled pool size for training: {len(X_unlabeled)}")
    print(f"Validation set size: {len(val_indices)}")

    # --- Model Definition ---
    model_params = {
        'd_model': args.d_model,
        'n_layers': args.n_layers,
        'dropout': args.dropout,
        'lr': args.lr,
        'encoder_lr': args.lr,
        'linear_lr': args.lr,
        'instrument_embedding_dim': args.instrument_embedding_dim
    }

    # Use the wrapper for the self-training classifier
    base_classifier = SelfTrainingPyTorchWrapper(
        model_class=SimpleSpectraTransformer,
        model_params=model_params,
        dataset=combined_dataset,
        batch_size=args.batch_size,
        epochs=args.inner_loop_epochs,
        num_workers=args.num_workers,
        accelerator=args.accelerator,
        devices=args.devices
    )

    self_training_model = SelfTrainingClassifier(
        base_classifier,
        threshold=args.confidence_threshold,
        max_iter=args.self_training_iterations,
        verbose=True
    )

    # --- Training Loop ---
    # We need to manually construct the initial training data for the first `fit` call.
    # sklearn's SelfTrainingClassifier expects `fit(X, y)`.
    # X will be indices for labeled data and a placeholder for unlabeled data.
    # y will have labels for the labeled data and -1 for unlabeled.
    X_initial_fit = np.concatenate([initial_labeled_indices, train_unlabeled_indices]).reshape(-1, 1)
    y_initial_fit = np.concatenate([y_initial_labeled, np.full(len(train_unlabeled_indices), -1)])

    # We need to pass validation data through to the wrapper.
    # We can't do it directly in sklearn's `fit`. So we patch the base classifier
    # to receive the validation indices and the logger.
    original_fit = base_classifier.fit
    base_classifier.fit = lambda X, y: original_fit(X, y, val_indices=val_indices, logger=csv_logger)

    print("\nStarting self-training process...")
    start_time = time.time()
    self_training_model.fit(X_initial_fit, y_initial_fit)
    end_time = time.time()
    print(f"Self-training finished in {end_time - start_time:.2f} seconds.")

    # --- Final Evaluation ---
    # Use the trained model to predict on the validation set
    print("\n--- Final Evaluation on Hold-out Validation Set ---")
    final_val_probs = self_training_model.predict_proba(val_indices.reshape(-1, 1))[:, 1]
    final_val_preds = (final_val_probs > 0.5).astype(int)
    final_val_labels = np.array([combined_dataset[i]['label'] for i in val_indices])

    accuracy = accuracy_score(final_val_labels, final_val_preds)
    precision = precision_score(final_val_labels, final_val_preds)
    recall = recall_score(final_val_labels, final_val_preds)
    f1 = f1_score(final_val_labels, final_val_preds)

    print(f"Final Validation Accuracy: {accuracy:.4f}")
    print(f"Final Validation Precision: {precision:.4f}")
    print(f"Final Validation Recall: {recall:.4f}")
    print(f"Final Validation F1-Score: {f1:.4f}")

    csv_logger.log_metrics({
        'final_val_accuracy': accuracy,
        'final_val_precision': precision,
        'final_val_recall': recall,
        'final_val_f1': f1
    })
    csv_logger.save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Self-training script for mass spec data.")

    # Data args
    parser.add_argument("--file_paths", type=str, required=True, help="Path to file containing mzML and CSV paths")
    parser.add_argument("--log_dir", type=str, default="./logs_self_training", help="Directory for logs")
    parser.add_argument("--val_split", type=float, default=0.2,
                        help="Proportion of the unlabeled pool to use for validation.")
    parser.add_argument("--unlabeled_pos_proportion", type=float, default=0.95,
                        help="Proportion of positive samples to treat as unlabeled.")

    # Model args
    parser.add_argument("--d_model", type=int, default=64, help="Model dimension")
    parser.add_argument("--n_layers", type=int, default=2, help="Number of transformer layers")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--instrument_embedding_dim", type=int, default=16,
                        help="Dimension of the instrument embedding output")

    # Training args
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size for training the inner model.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of DataLoader workers.")
    parser.add_argument("--inner_loop_epochs", type=int, default=50,
                        help="Epochs for each training iteration within self-training.")
    parser.add_argument("--self_training_iterations", type=int, default=2,
                        help="Max iterations for the self-training loop.")
    parser.add_argument("--confidence_threshold", type=float, default=0.7,
                        help="Confidence threshold to add pseudo-labels.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed.")

    # Cluster/Device args
    parser.add_argument("--accelerator", type=str, default="gpu", help="Accelerator to use ('gpu', 'cpu')")
    parser.add_argument("--devices", type=int, default=1, help="Number of devices to use.")

    args = parser.parse_args()
    main(args)