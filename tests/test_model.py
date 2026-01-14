# tests/test_model.py
import torch
import pytest

from mnist.model import MyAwesomeModel


def test_model_output_shape():
    model = MyAwesomeModel()
    x = torch.randn(1, 1, 28, 28)
    y = model(x)

    assert y.shape == (1, 10), "Model output shape incorrect"


def test_error_on_wrong_shape():
    model = MyAwesomeModel()

    with pytest.raises(ValueError):
        model(torch.randn(1, 2, 28, 28))