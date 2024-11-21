import torch
import lightning.pytorch as pl
from torch.utils.data import DataLoader, TensorDataset
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.loggers import CSVLogger

class SimpleModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(10, 1)
        self.loss_fn = torch.nn.MSELoss()

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        print(f"Epoch {self.current_epoch}, Batch {batch_idx}, Loss: {loss.item()}")
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

# Generate dummy data
x = torch.randn(100, 10)
y = torch.randn(100, 1)
dataset = TensorDataset(x, y)
dataloader = DataLoader(dataset, batch_size=32)

# Set up logger
csv_logger = CSVLogger("csv_logs", name="simple_test_csv")

# Train
trainer = pl.Trainer(max_epochs=5, logger=csv_logger)
model = SimpleModel()
trainer.fit(model, dataloader)
