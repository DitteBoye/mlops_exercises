import os
import pytest
import torch

from mnist.data import corrupt_mnist
from tests import _PATH_DATA

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

@pytest.mark.skipif(
    not os.path.exists(
        os.path.join(_PATH_DATA, "processed", "train_images.pt")
    ),
    reason="Processed data not found",
)
def test_data():
    train, test = corrupt_mnist()

    assert len(train) == 30000
    assert len(test) == 5000
