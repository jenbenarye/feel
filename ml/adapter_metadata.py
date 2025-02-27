from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict
import json

@dataclass
class AdapterMetadata:
    """Metadata for tracking adapter training history"""
    training_timestamp: str  # ISO format timestamp
    training_params: Dict  # Training parameters used
    model_name: str  # Base model name
    language: str  # Language of the adapter
    version: str  # Version of the adapter

    # Create class instance from a dictionary
    @classmethod
    def from_dict(cls, data: Dict):
        return cls(**data)

    # Convert class instance to a dictionary
    def to_dict(self) -> Dict:
        return {
            "training_timestamp": self.training_timestamp,
            "dataset_entries": self.dataset_entries,
            "training_params": self.training_params,
            "model_name": self.model_name,
            "language": self.language,
            "version": self.version
        }

    # Save metadata to a JSON file
    def save(self, filepath: str):
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    # Load metadata from a JSON file
    @classmethod
    def load(cls, filepath: str):
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
