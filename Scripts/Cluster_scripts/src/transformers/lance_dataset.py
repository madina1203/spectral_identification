import torch
import torch.nn as nn
import pyarrow as pa
from depthcharge.data import SpectrumDataset

class CustomSpectrumDataset(SpectrumDataset):
    """
    A custom dataset subclass based on SpectrumDataset that reads data from a Lance file
    and additionally returns extra entries such as instrument settings, label, and precursor m/z.
    """

    def _to_tensor(self, batch: pa.RecordBatch) -> dict[str, torch.Tensor]:
        """
        Convert a record batch from the Lance file to PyTorch tensors.
        This method extends the base _to_tensor() method from SpectrumDataset.

        Parameters
        ----------
        batch : pa.RecordBatch
            A batch of records read from the Lance dataset.

        Returns
        -------
        dict
            A dictionary where variable-length sequences (e.g., m/z and intensity arrays)
            are padded and additional fields are converted to tensors.
        """
        # Call the parent method to process the basic fields (e.g. pad m/z and intensity arrays).
        data = super()._to_tensor(batch)

        # Process instrument_settings:
        if "instrument_settings" in data:
            # If instrument_settings is a list (one per sample), stack them into a tensor.
            if isinstance(data["instrument_settings"], list):
                try:
                    # Try stacking assuming each element is already a tensor or convertible.
                    data["instrument_settings"] = torch.stack([
                        x if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.float32)
                        for x in data["instrument_settings"]
                    ])
                except Exception:
                    data["instrument_settings"] = data["instrument_settings"].clone().detach().to(torch.float32)
            else:
                # Check if instrument_settings is already a tensor
                if isinstance(data["instrument_settings"], torch.Tensor):
                    data["instrument_settings"] = data["instrument_settings"].clone().detach().to(torch.float32)
                else:
                    data["instrument_settings"] = torch.tensor(data["instrument_settings"], dtype=torch.float32)

        # Process label:
        if "label" in data:
            # In case label is a list of scalars
            if isinstance(data["label"], list):
                data["label"] = torch.tensor(data["label"], dtype=torch.float32)
            else:
                # Check if label is already a tensor
                if isinstance(data["label"], torch.Tensor):
                    data["label"] = data["label"].clone().detach().to(torch.float32)
                else:
                    data["label"] = torch.tensor([data["label"]], dtype=torch.float32)

        # Process precursor_mz:
        if "precursor_mz" in data:
            if isinstance(data["precursor_mz"], list):
                data["precursor_mz"] = torch.tensor(data["precursor_mz"], dtype=torch.float32)
            else:
                # Check if precursor_mz is already a tensor
                if isinstance(data["precursor_mz"], torch.Tensor):
                    data["precursor_mz"] = data["precursor_mz"].clone().detach().to(torch.float32)
                else:
                    data["precursor_mz"] = torch.tensor([data["precursor_mz"]], dtype=torch.float32)

        # Return the processed batch
        return data

    def __len__(self):
        # Assume that the parent SpectrumDataset defines a property `n_spectra`
        # that returns the total number of spectra.
        return self.n_spectra

    @classmethod
    def from_lance(
        cls,
        path: str,
        batch_size: int,
        parse_kwargs: dict | None = None,
        **kwargs,
    ) -> "CustomSpectrumDataset":
        """
        Create a CustomSpectrumDataset instance from an existing Lance file.

        Parameters
        ----------
        path : str
            The path to the Lance dataset (e.g. '/path/to/dataset.lance').
        batch_size : int
            The batch size (this is independent of the PyTorch DataLoader batch size).
        parse_kwargs : dict, optional
            Additional keyword arguments for parsing.
        **kwargs : dict
            Any additional keyword arguments for the parent class.

        Returns
        -------
        CustomSpectrumDataset
            An instance of the custom dataset ready for training.
        """
        # Use the parent's from_lance method to load the dataset,
        # but instantiate our custom class instead.
        return cls(
            spectra=None,
            batch_size=batch_size,
            path=path,
            parse_kwargs=parse_kwargs,
            **kwargs,
        )
