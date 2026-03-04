"""
Standard dataset registrations for ConTSG.

This module registers all standard datasets that use the default TimeSeriesDataset
loader. Each dataset expects the standard file structure:

    data_folder/
    ├── meta.json              # Dataset metadata
    ├── train_ts.npy          # Training time series (N, L, C)
    ├── train_cap_emb.npy     # Training caption embeddings (N, D)
    ├── train_attrs_idx.npy   # Training attributes (N, A) [optional]
    ├── valid_ts.npy
    ├── valid_cap_emb.npy
    └── test_ts.npy, test_cap_emb.npy

All datasets use BaseDataModule's default implementation without customization.
"""

from contsg.data.datamodule import BaseDataModule
from contsg.registry import Registry


# Standard datasets with descriptions
# Format: {registry_name: docstring}
_STANDARD_DATASETS = {
    # Synthetic datasets
    "synth-m": "Synthetic multivariable dataset with various patterns (sine, triangle, square).",
    "synth-u": "Synthetic univariable dataset with diverse patterns and natural language descriptions.",
    # Real-world datasets
    "ettm1": "Electricity Transformer Temperature (ETTm1).",
    "weather_concept": "Weather with concept-level (semantic) text/attribute conditions.",
    "weather_morphology": "Weather with morphology-level (structural) text/attribute conditions.",
    "istanbul_traffic": "Istanbul traffic flow measurements from road network.",
    "airquality_beijing": "Beijing air quality dataset with attribute conditions.",
    # PTB-XL ECG datasets
    "ptbxl_concept": "PTB-XL ECG dataset with concept-level conditions.",
    "ptbxl_morphology": "PTB-XL ECG dataset with morphology-level conditions.",
}


def _create_dataset_class(name: str, docstring: str) -> type:
    """Create a dataset class with the given name and docstring."""
    class_name = "".join(word.title() for word in name.replace("-", "_").split("_")) + "DataModule"
    return type(class_name, (BaseDataModule,), {"__doc__": docstring})


# Register all standard datasets
for _name, _doc in _STANDARD_DATASETS.items():
    Registry.register_dataset(_name)(_create_dataset_class(_name, _doc))


# Clean up module namespace
del _name, _doc
