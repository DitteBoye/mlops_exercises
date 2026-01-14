import torch
import typer
import wandb

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from .data import corrupt_mnist
from .model import MyAwesomeModel


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


def evaluate(model_path: str = "model.pth"):
    run = wandb.init(project="corrupt_mnist", job_type="evaluation")

    model = MyAwesomeModel().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    _, test_set = corrupt_mnist()
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=64)

    preds, targets = [], []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(DEVICE)
            logits = model(x)

            preds.append(logits.argmax(dim=1).cpu())
            targets.append(y)

    preds = torch.cat(preds).numpy()
    targets = torch.cat(targets).numpy()

    metrics = {
        "accuracy": accuracy_score(targets, preds),
        "precision": precision_score(targets, preds, average="weighted"),
        "recall": recall_score(targets, preds, average="weighted"),
        "f1": f1_score(targets, preds, average="weighted"),
    }

    wandb.log(metrics)
    print(metrics)

    run.finish()


def main():
    typer.run(evaluate)
