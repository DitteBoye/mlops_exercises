# tests/test_data.py
import os
import pytest
import torch

from mnist.data import corrupt_mnist
from tests import _PATH_DATA


# @pytest.mark.skipif(
#     not os.path.exists(os.path.join(_PATH_DATA, "processed")),
#     reason="Processed data not found",
# )
def test_data():
    train, test = corrupt_mnist()

    assert len(train) == 30000, "Train set size incorrect"
    assert len(test) == 5000, "Test set size incorrect"

    for dataset in [train, test]:
        for x, y in dataset:
            assert x.shape == (1, 28, 28), "Image shape incorrect"
            assert y in range(10), "Label out of range"

    train_targets = torch.unique(train.tensors[1])
    test_targets = torch.unique(test.tensors[1])

    assert (train_targets == torch.arange(10)).all()
    assert (test_targets == torch.arange(10)).all()
