import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import sys
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

# import pulearn
from pulearn import ElkanotoPuClassifier, WeightedElkanotoPuClassifier


from src.transformers.simpleDataset import SimpleMassSpecDataset
from src.transformers.model_simplified import SimpleSpectraTransformer


SEED = 1
pl.seed_everything(SEED, workers=True)

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
    
    def __init__(self, model=None, dataset=None, val_dataset=None, d_model=64, n_layers=2,
                 dropout=0.3, lr=0.001, batch_size=64, epochs=5, device=None, num_workers=4,
                 instrument_embedding_dim=32, force_cpu=False):
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
                device : str or None
                    Computing device ('cpu', 'cuda', 'mps').
                num_workers : int
                    Number of worker processes for data loading.
                instrument_embedding_dim : int
                    Dimension of the instrument embedding.
        """
        self.model = model
        self.dataset = dataset
        self.val_dataset = val_dataset
        self.d_model = d_model
        self.n_layers = n_layers
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.num_workers = num_workers
        self.instrument_embedding_dim = instrument_embedding_dim
        
        self.force_cpu = force_cpu
        
        # Force CPU usage if specified
        if force_cpu:
            self.device = torch.device("cpu")
        elif device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else
                                      "mps" if torch.backends.mps.is_available() else
                                      "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        self.is_fitted_ = False
        self.classes_ = np.array([-1.0, 1.0])
    
    def fit(self, X, y, sample_weight=None):
        """
       Model training.
        
        Parameters:
        -----------
        X : array-like, shape = [n_samples, n_features]
            Indices of training data.
        y : array-like, shape = [n_samples]
            Class labels.
        sample_weight : array-like, optional
            Sample weights.
            
        Returns:
        --------
        self : object
            Returns self.
        """
        # Create model if not provided
        if self.model is None:
            self.model = SimpleSpectraTransformer(
                d_model=self.d_model,
                n_layers=self.n_layers,
                dropout=self.dropout,
                lr=self.lr,
                instrument_embedding_dim=self.instrument_embedding_dim
            )
        
        # Move the model and all its components to the appropriate device
        self.model = self.model.to(self.device)
        
        # Ensure all model components are on the same device
        if hasattr(self.model, 'spectrum_encoder'):
            self.model.spectrum_encoder = self.model.spectrum_encoder.to(self.device)
        
        # Get indices from X
        indices = X.flatten()
        print("the length of indices is :", len(indices))
        # Create a subset of the dataset using indices from X
        if self.dataset is None:
            raise ValueError("Dataset must be provided")
        
        # For ElkanotoPuClassifier we use all examples, not just positive ones
        train_dataset = Subset(self.dataset, indices)
        
        # Create DataLoader with optimized settings for cluster GPUs
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            persistent_workers=True,
            pin_memory=True
        )
        
        # Create validation DataLoader if val_dataset is provided
        val_loader = None
        if self.val_dataset is not None:
            val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                collate_fn=collate_fn,
                persistent_workers=True,
                pin_memory=True,
                shuffle=False
            )
        
        # Create PyTorch Lightning trainer with DDP strategy
        trainer = pl.Trainer(
            max_epochs=self.epochs,
            logger=None,
            accelerator="gpu",
            devices=2,  # Use 2 GPUs
            strategy=DDPStrategy(gradient_as_bucket_view=True, static_graph=True),
            precision="16-mixed",  # Use mixed precision for V100 GPUs
        )
        
        # Train the model
        if val_loader:
            trainer.fit(self.model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        else:
            trainer.fit(self.model, train_dataloaders=train_loader)
        
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
            persistent_workers=True,
            pin_memory=True,
            shuffle=False
        )
        
        all_probs = []
        with torch.no_grad():
            for batch in test_loader:
                # Move data to the appropriate device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                # Get predictions from the model
                try:
                    # Explicitly move all tensors to the model's device
                    for k, v in batch.items():
                        if isinstance(v, torch.Tensor):
                            batch[k] = v.to(self.device)
                    
                    # Make sure the model and all its components are on the right device
                    self.model = self.model.to(self.device)
                    if hasattr(self.model, 'spectrum_encoder'):
                        self.model.spectrum_encoder = self.model.spectrum_encoder.to(self.device)
                    
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
        # scikit-learn expects probabilities for both classes: [P(y=0), P(y=1)]
        return np.column_stack([1 - probs_numpy, probs_numpy])
    
    def predict(self, X, threshold=0.5, sample_weight=None):
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

def main(args):
    # Set up logger
    csv_logger = CSVLogger(args.log_dir, name="pu_learning_experiment_3")
    
    # Print GPU information
    if torch.cuda.is_available():
        print(f"Available GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Read file paths from file_paths_local_full.txt
    with open(args.file_paths, 'r') as f:
        lines = f.readlines()
    mzml_files = [line.split(",")[0].strip() for line in lines]
    csv_files = [line.split(",")[1].strip() for line in lines]
    
    # Load data from mzML and CSV files
    print("Loading data from mzML and CSV files...")
    datasets = [SimpleMassSpecDataset(mzml, csv) for mzml, csv in zip(mzml_files, csv_files)]
    combined_dataset = ConcatDataset(datasets)
    print("Length of dataset:", len(combined_dataset))
    
    # Extract labels for all examples
    labels = [combined_dataset[i]['label'] for i in range(len(combined_dataset))]
    labels = np.array(labels, dtype=np.int64)
    
    # Split data into training and validation sets
    indices = np.arange(len(combined_dataset))
    train_indices, val_indices = train_test_split(
        indices, test_size=args.val_split, random_state=SEED
    )
    
    # Create subsets for training and validation
    train_dataset = Subset(combined_dataset, train_indices)
    val_dataset = Subset(combined_dataset, val_indices)
    
    print(f"Total samples: {len(combined_dataset)}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Extract labels for the training set
    train_labels = [combined_dataset[i]['label'] for i in train_indices]

    # Count positive and negative samples for the weighted classifier
    num_positive = np.sum(train_labels == 1)
    num_unlabeled = np.sum(train_labels == 0)

    # Transform labels for PU learning: 1.0 for positive, -1.0 for unlabeled examples
    pu_labels = np.array([1.0 if label == 1 else -1.0 for label in train_labels])
    
    print(f"Number of positive samples in training set: {np.sum(pu_labels == 1.0)}")
    print(f"Number of unlabeled samples in training set: {np.sum(pu_labels == -1.0)}")
    
    # Create PyTorch model with optimized parameters for cluster GPUs
    model = SimpleSpectraTransformer(
        d_model=args.d_model,
        n_layers=args.n_layers,
        dropout=args.dropout,
        lr=args.lr,
        instrument_embedding_dim=args.instrument_embedding_dim
    )
    
    # Create wrapper for PyTorch model
    pytorch_model = ImprovedPyTorchSklearnWrapper(
        model=model,
        dataset=combined_dataset,
        val_dataset=val_dataset,
        d_model=args.d_model,
        n_layers=args.n_layers,
        dropout=args.dropout,
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        num_workers=args.num_workers,
        instrument_embedding_dim=args.instrument_embedding_dim,
        force_cpu=args.force_cpu
    )
    
    # Create PU classifier with our model wrapper
    # pu_estimator = ElkanotoPuClassifier(
    #     estimator=pytorch_model,
    #     hold_out_ratio=args.hold_out_ratio
    # )
    #Weighted Elkanato
    pu_estimator = WeightedElkanotoPuClassifier(
        estimator=pytorch_model,  #
        labeled=num_positive,  # Number of positive samples
        unlabeled=num_unlabeled,  # Number of unlabeled samples
        hold_out_ratio=args.hold_out_ratio
    )
    # Print information about the classifier
    print(f"PU classifier: {pu_estimator}")
    print(f"Base classifier: {pytorch_model}")
    print(f"Model device: {next(pytorch_model.model.parameters()).device}")
    
    print("Starting training of the PU classifier...")
    
    # Convert training indices to an array (each sample index in its own row)
    X_indices = np.array(train_indices).reshape(-1, 1)
    
    print("Converting data for the PU classifier...")
    print(f"X_indices shape: {X_indices.shape}")
    print(f"pu_labels shape: {pu_labels.shape}")
    
    # Train the PU classifier using indices and transformed labels
    try:
        pu_estimator.fit(X_indices, pu_labels)
    except Exception as e:
        print(f"Error during PU classifier training: {e}")
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
            val_dataset=val_dataset,
            d_model=args.d_model,
            n_layers=args.n_layers,
            dropout=args.dropout,
            lr=args.lr,
            batch_size=args.batch_size,
            epochs=args.epochs,
            num_workers=args.num_workers,
            instrument_embedding_dim=args.instrument_embedding_dim,
            force_cpu=True
        )
        
        # Create a new PU classifier
        pu_estimator = ElkanotoPuClassifier(
            estimator=pytorch_model,
            hold_out_ratio=args.hold_out_ratio
        )
        
        # Try training again
        pu_estimator.fit(X_indices, pu_labels)
    
    print("PU classifier training completed.")
    
    # Predictions on the validation set
    print("Making predictions on the validation set...")
    X_val_indices = np.array(val_indices).reshape(-1, 1)
    val_predictions = pu_estimator.predict(X_val_indices)
    # Convert predicted labels to PU format (-1.0 for unlabeled, 1.0 for positive)
    val_predictions = np.array([1.0 if p == 1.0 else -1.0 for p in val_predictions])

    # Get true labels for the validation set and convert to PU format
    val_true_labels = np.array([combined_dataset[i]['label'] for i in val_indices])
    val_true_labels_pu = np.array([1.0 if label == 1 else -1.0 for label in val_true_labels])
    
    # Calculate evaluation metrics using scikit-learn
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    # Check unique values in labels
    unique_true = np.unique(val_true_labels_pu)
    unique_pred = np.unique(val_predictions)
    print(f"Unique values in true labels: {unique_true}")
    print(f"Unique values in predicted labels: {unique_pred}")
    
    # If there are more than two classes, use 'macro' as average
    if len(unique_true) > 2 or len(unique_pred) > 2:
        average = 'macro'
        print(f"More than two classes detected, using average='{average}'")
    else:
        average = 'binary'
    
    accuracy = accuracy_score(val_true_labels_pu, val_predictions)
    precision = precision_score(val_true_labels_pu, val_predictions, average=average)
    recall = recall_score(val_true_labels_pu, val_predictions, average=average)
    f1 = f1_score(val_true_labels_pu, val_predictions, average=average)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Save the model
    model_save_path = os.path.join(args.log_dir, "pu_model.pt")
    torch.save(pytorch_model.model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
    
    # Save the model
    model_save_path = os.path.join(args.log_dir, "pu_model.pt")
    torch.save(pytorch_model.model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_paths", type=str, required=True, help="Path to file containing mzML and CSV paths")
    parser.add_argument("--log_dir", type=str, default="./logs", help="Directory for logs")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
    parser.add_argument("--val_split", type=float, default=0.3, help="Validation split")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of DataLoader workers")
    parser.add_argument("--d_model", type=int, default=64, help="Model dimension")
    parser.add_argument("--n_layers", type=int, default=6, help="Number of transformer layers")
    parser.add_argument("--dropout", type=float, default=0.135378967114, help="Dropout rate")
    parser.add_argument("--lr", type=float, default=0.0002269876583, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--hold_out_ratio", type=float, default=0.1, help="Hold-out ratio for PU learning")
    parser.add_argument("--instrument_embedding_dim", type=int, default=16, help="Dimension of the instrument embedding output")
    parser.add_argument("--force_cpu", action="store_true", help="Force using CPU instead of GPU/MPS")
    args = parser.parse_args()
    main(args)
