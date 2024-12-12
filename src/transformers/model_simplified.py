import torch
import torch.nn as nn
import lightning.pytorch as pl
from depthcharge.transformers import SpectrumTransformerEncoder  # Import DepthCharge transformer
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision
from torchmetrics.classification import BinaryF1Score, BinaryRecall
from depthcharge.encoders import FloatEncoder


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

            # Ensure precursor_mz has shape (batch_size, 1)
        precursor_mz = precursor_mz.type_as(mz_array).view(-1, 1)
        # print(f"precursor_mz shape: {precursor_mz.shape}")  # Debug statement

            # Encode the precursor m/z
        mz_rep = self.precursor_mz_encoder(precursor_mz)
        # print(f"mz_rep shape before squeeze: {mz_rep.shape}")  # Debug statement

            # Squeeze the sequence length dimension if necessary
        if mz_rep.dim() == 3 and mz_rep.shape[1] == 1:
            mz_rep = mz_rep.squeeze(1)
            # print(f"mz_rep shape after squeeze: {mz_rep.shape}")  # Debug statement

            # Now mz_rep should have shape (batch_size, d_model)
        return mz_rep  # Remove batch dimension and return

    def init_weights(self, module):
        """Custom weight initialization logic."""
        if isinstance(module, nn.Linear):  # Initialize Linear layers
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):  # Initialize LayerNorm layers
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):  # Initialize embeddings if used
            nn.init.normal_(module.weight, mean=0, std=1)


class SimpleSpectraTransformer(pl.LightningModule):
    """Simplified model using only SpectrumTransformerEncoder for MS1 spectra."""

    def __init__(
            self,
            d_model,
            n_layers,
            dropout=0.1,
            lr=0.001
    ):
        """Initialize the model with transformer for spectra only."""
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.lr = lr


        # Transformer encoder for spectra data from DepthCharge
        self.spectrum_encoder = MyModel(
            d_model=d_model,
            n_layers=n_layers,
            dropout=dropout,

        )
        #adding complexity by 1 linear layer
        self.fc_instrument_1 = nn.Linear(20, d_model)
        # self.fc_instrument_2 = nn.Linear(d_model, d_model)
        # Fully connected layers for binary classification
        self.fc_combined = nn.Linear(2 * d_model, d_model)
        self.fc_output = nn.Linear(d_model, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()  # Binary classification

        class_weights = torch.tensor([1 - 0.17, 0.17], dtype=torch.float32)  # Adjust weights based on proportions

        #self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=class_weights[1])
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
        # Apply He initialization to all relevant layers
        self.apply(self.init_weights)

    def init_weights(self, module):
        """Apply He initialization to layers."""
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    def forward(self, batch):
        """Forward pass through the transformer and classification layers."""
        # Spectra data (m/z and intensity) processing with SpectrumTransformerEncoder
        mz_array = batch["mz"].float()
        intensity_array = batch["intensity"].float()
        precursor_mz = batch["precursor_mz"].float()  # Get precursor m/z from the batch

        # Forward pass through DepthCharge Transformer for spectra
        spectra_emb, _ = self.spectrum_encoder(mz_array, intensity_array, precursor_mz=precursor_mz)
        spectra_emb = spectra_emb[:, 0, :]  # Get the global embedding (first token)
        # Instrument settings data processing with fully connected layers simple version
        instrument_settings = batch["instrument_settings"].float()  # Shape: (batch_size, 27)
        print("First 10 instrument settings:", instrument_settings[1, :10])
        print("Last 10 instrument settings:", instrument_settings[1, -10:])

        instrument_emb = self.fc_instrument_1(instrument_settings)
        instrument_emb = self.relu(instrument_emb)


        # Add debug statements to check the magnitude of embeddings
        print("Spectra embedding magnitude:", spectra_emb.abs().mean().item())
        print("Instrument embedding magnitude:", instrument_emb.abs().mean().item())
        # instrument_emb = self.fc_instrument_2(instrument_emb)
        # instrument_emb = self.relu(instrument_emb)
        # Combine embeddings from spectra and instrument settings
        combined_emb = torch.cat((spectra_emb, instrument_emb), dim=-1)
        print(combined_emb[0,:10])
        print(combined_emb[0,-10:])
        combined_emb=self.fc_combined(combined_emb)
        combined_emb = self.relu(combined_emb)
        # quit()

        # Classification layers
        # output = self.fc_output(spectra_emb)
        output = self.fc_output(combined_emb)
        return output

    def step(self, batch):
        """Calculate loss and perform the forward pass."""
        logits = self(batch)
        target = batch["labels"].int()
        loss = self.bce_loss(logits.view(-1, 1), target.view(-1, 1).float())

        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).int()
        if target.numel() == 0 or preds.numel() == 0:
            raise ValueError("Empty predictions or targets in batch.")
        return loss, preds, target

    def training_step(self, batch, batch_idx):
        """A single training step."""
        loss, preds, targets = self.step(batch)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        # Update metrics
        self.train_accuracy.update(preds, targets)
        self.train_precision.update(preds, targets)
        self.train_f1.update(preds, targets)
        self.train_recall.update(preds, targets)

        # Log metrics
        # self.log("train_accuracy", self.train_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        # self.log("train_precision", self.train_precision, on_step=False, on_epoch=True)
        # self.log("train_f1", self.train_f1, on_step=False, on_epoch=True, prog_bar=True)
        # self.log("train_recall", self.train_recall, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """A single validation step."""
        loss, preds, targets = self.step(batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        # Update metrics
        self.val_accuracy.update(preds, targets)
        self.val_precision.update(preds, targets)
        self.val_f1.update(preds, targets)
        self.val_recall.update(preds, targets)

        # Log metrics
        # self.log("val_accuracy", self.val_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        # self.log("val_precision", self.val_precision, on_step=False, on_epoch=True)
        # self.log("val_f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=True)
        # self.log("val_recall", self.val_recall, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def on_train_epoch_end(self):
        """Reset training metrics at the end of each epoch."""
        self.log("train_accuracy", self.train_accuracy.compute(), prog_bar=True)
        self.log("train_precision", self.train_precision.compute())
        self.log("train_f1", self.train_f1.compute(), prog_bar=True)
        self.log("train_recall", self.train_recall.compute(), prog_bar=True)
        self.train_accuracy.reset()
        self.train_precision.reset()
        self.train_f1.reset()
        self.train_recall.reset()

    def on_validation_epoch_end(self):
        """Reset validation metrics at the end of each epoch."""
        self.log("val_accuracy", self.val_accuracy.compute(), prog_bar=True)
        self.log("val_precision", self.val_precision.compute())
        self.log("val_f1", self.val_f1.compute(), prog_bar=True)
        self.log("val_recall", self.val_recall.compute(), prog_bar=True)

        self.val_accuracy.reset()
        self.val_precision.reset()
        self.val_f1.reset()
        self.val_recall.reset()

    def configure_optimizers(self):
        """Configure optimizer for training."""
        # optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        # trying different optimizers
        optimizer = torch.optim.AdamW(
            [
                {"params": self.spectrum_encoder.parameters(), "lr": 0.0001, "weight_decay": 0.01},  # Transformer part
                {"params": self.fc_output.parameters(), "lr": 0.0001, "weight_decay": 0.01},  # Linear layer
                {"params": self.fc_combined.parameters(), "lr": 0.001, "weight_decay": 0.01},  # Another linear layer
                {"params": self.fc_instrument_1.parameters(), "lr": 0.001, "weight_decay": 0.01}
            ]

        )
        return optimizer
