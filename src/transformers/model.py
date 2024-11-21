import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from depthcharge.transformers import SpectrumTransformerEncoder  # Import DepthCharge transformer
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision
from torchmetrics.classification import BinaryF1Score, BinaryRecall



class SpectraTransformerWithInstrumentSettings(pl.LightningModule):
    """Model with SpectrumTransformerEncoder for MS1 spectra and linear layers for instrument settings."""

    def __init__(
            self,
            d_model,
            n_layers,
            dropout=0.1,
            lr=0.001
    ):
        """Initialize the model with transformer for spectra and FC layers for instrument settings."""
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.lr = lr

        # Transformer encoder for spectra data from DepthCharge
        self.spectrum_encoder = SpectrumTransformerEncoder(
            d_model=d_model,
            n_layers=n_layers,
            dropout=dropout,
        )

        # Fully connected layers for instrument settings
        self.fc_instrument_1 = nn.Linear(27, d_model)  # First FC layer for instrument settings (24-> 64)
        #removing for now (simplify)
        # self.fc_instrument_2 = nn.Linear(64, d_model)  # Second FC layer (64 -> d_model)
        self.fc_instrument_2 = nn.Linear(d_model, d_model)  # Second FC layer (reducing dimensionality)
        self.bn_instrument = nn.BatchNorm1d(d_model)  # Add BatchNorm for instrument settings
        self.dropout = nn.Dropout(dropout)  # Define Dropout layer here

        # Fully connected layers for combining embeddings and binary classification
        # self.fc_combined = nn.Linear(2 * d_model, d_model)  # Combine spectra and instrument settings embeddings
        self.fc_combined = nn.Linear(2 * d_model, 2 * d_model)  # Increase capacity
        self.fc_combined_out = nn.Linear(2 * d_model, d_model)  # Reduce back to d_model
        self.fc_output = nn.Linear(d_model, 1)  # Final output layer for binary classification
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()  # Binary classification

        # Loss function
        # self.bce_loss = nn.BCELoss()  # Binary Cross-Entropy Loss
        self.bce_loss = nn.BCEWithLogitsLoss()
        # Define metrics for training and validation
        self.train_accuracy = BinaryAccuracy()
        self.train_precision = BinaryPrecision()
        self.val_accuracy = BinaryAccuracy()
        self.val_precision = BinaryPrecision()
        self.train_f1 = BinaryF1Score()
        self.train_recall = BinaryRecall()
        self.val_f1 = BinaryF1Score()
        self.val_recall = BinaryRecall()

    def forward(self, batch):
        """Forward pass through the transformer and fully connected layers."""

        # Spectra data (m/z and intensity) processing with SpectrumTransformerEncoder
        mz_array = batch["mz"].float()
        intensity_array = batch["intensity"].float()

        # Forward pass through DepthCharge Transformer for spectra
        spectra_emb, _ = self.spectrum_encoder(mz_array, intensity_array)
        spectra_emb = spectra_emb[:, 0, :]  # Get the global embedding (first token)

        # Instrument settings data processing with fully connected layers simple version
        instrument_settings = batch["instrument_settings"].float()  # Shape: (batch_size, 27)
        instrument_emb = self.fc_instrument_1(instrument_settings)
        instrument_emb = self.bn_instrument(instrument_emb)  # Normalize after first FC

        instrument_emb = self.relu(instrument_emb)
        instrument_emb = self.dropout(instrument_emb)  # Apply dropout

        instrument_emb = self.fc_instrument_2(instrument_emb)
        instrument_emb = self.relu(instrument_emb)  # Added ReLU
        instrument_emb = self.dropout(instrument_emb)  # Apply dropout

        # Combine embeddings from spectra and instrument settings
        combined_emb = torch.cat((spectra_emb, instrument_emb), dim=-1)

        # Final classification layers
        # combined_emb = self.fc_combined(combined_emb)
        # combined_emb = self.relu(combined_emb)  # Added ReLU
        combined_emb = self.fc_combined(combined_emb)
        combined_emb = self.relu(combined_emb)
        combined_emb = self.fc_combined_out(combined_emb)
        combined_emb = self.relu(combined_emb)

        # output = self.sigmoid(self.fc_output(combined_emb))  # Binary classification (0 or 1)
        output = self.fc_output(combined_emb)
        print(f"spectra_emb shape: {spectra_emb.shape}")
        print(f"instrument_emb shape: {instrument_emb.shape}")
        print(f"combined_emb shape: {combined_emb.shape}")
        return output

    def step(self, batch):
        """Calculate loss and perform the forward pass."""
        # pred = self(batch)
        # target = batch["labels"].float()
        # loss = self.bce_loss(pred, target.view(-1, 1))
        logits = self(batch)
        target = batch["labels"].float()
        loss = self.bce_loss(logits.view(-1, 1), target.view(-1, 1))
        preds = torch.sigmoid(logits)  # Apply sigmoid to logits for metrics
        # print(f"Predictions: {preds}")
        # print(f"Targets: {target}")
        return loss, preds, target
        # return loss, pred, target

    def training_step(self, batch, batch_idx):
        """A single training step."""
        loss, preds, targets = self.step(batch)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        # Update metrics
        self.train_accuracy.update(preds, targets)
        self.train_precision.update(preds, targets)
        # Update and log F1 and Recall metrics
        self.train_f1.update(preds, targets)
        self.train_recall.update(preds, targets)

        self.log("train_f1", self.train_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_recall", self.train_recall, on_step=False, on_epoch=True, prog_bar=True)

        self.log("train_accuracy", self.train_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_precision", self.train_precision, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """A single validation step."""
        loss, preds, targets = self.step(batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        # Update metrics
        self.val_accuracy.update(preds, targets)
        self.val_precision.update(preds, targets)
        # Update and log F1 and Recall metrics
        self.val_f1.update(preds, targets)
        self.val_recall.update(preds, targets)
        # Log metrics
        self.log("val_accuracy", self.val_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_precision", self.val_precision, on_step=False, on_epoch=True)
        self.log("val_f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_recall", self.val_recall, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        """Reset training metrics at the end of each epoch."""
        self.train_accuracy.reset()
        self.train_precision.reset()
        self.train_f1.reset()
        self.train_recall.reset()

    def on_validation_epoch_end(self):
        """Reset validation metrics at the end of each epoch."""
        self.val_accuracy.reset()
        self.val_precision.reset()
        self.val_f1.reset()
        self.val_recall.reset()

    def configure_optimizers(self):
        """Configure optimizer for training."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
