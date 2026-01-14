import torch
import typer
import wandb
import matplotlib.pyplot as plt
from sklearn.metrics import (
    RocCurveDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from .data import corrupt_mnist
from .model import MyAwesomeModel
from .logger import setup_logger


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


def train():
    logger = setup_logger()
    logger.info("Starting training")

    run = wandb.init(project="corrupt_mnist")
    config = wandb.config

    lr = config.lr
    batch_size = config.batch_size
    epochs = config.epochs

    model = MyAwesomeModel().to(DEVICE)
    train_set, _ = corrupt_mnist()
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True
    )

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    preds_all, targets_all = [], []

    for epoch in range(epochs):
        model.train()
        for i, (img, target) in enumerate(train_loader):
            img, target = img.to(DEVICE), target.to(DEVICE)

            optimizer.zero_grad()
            logits = model(img)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()

            acc = (logits.argmax(dim=1) == target).float().mean().item()

            wandb.log(
                {
                    "train_loss": loss.item(),
                    "train_accuracy": acc,
                    "epoch": epoch,
                }
            )

            preds_all.append(logits.detach().cpu())
            targets_all.append(target.detach().cpu())

            if i % 100 == 0:
                logger.info(f"Epoch {epoch} | Iter {i} | Loss {loss.item():.4f}")
                img_cpu = img.detach().cpu()
                images = [
                    wandb.Image(
                        (img_cpu[j] - img_cpu[j].min()) / (img_cpu[j].max() - img_cpu[j].min()),
                        caption=f"Image {j}",
                    )
                    for j in range(min(5, img_cpu.shape[0]))
                ]
                wandb.log({"images": images})

            grads = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
            wandb.log({"gradients": wandb.Histogram(grads.detach().cpu().numpy())})

    # ---------- final metrics ----------
    preds = torch.cat(preds_all)
    targets = torch.cat(targets_all)

    final_metrics = {
        "final_accuracy": accuracy_score(targets, preds.argmax(dim=1)),
        "final_precision": precision_score(
            targets, preds.argmax(dim=1), average="weighted"
        ),
        "final_recall": recall_score(
            targets, preds.argmax(dim=1), average="weighted"
        ),
        "final_f1": f1_score(
            targets, preds.argmax(dim=1), average="weighted"
        ),
    }

    wandb.log(final_metrics)

    # ---------- ROC curves ----------
    for class_id in range(10):
        one_hot = (targets == class_id).int()
        RocCurveDisplay.from_predictions(one_hot, preds[:, class_id])

    wandb.log({"roc": wandb.Image(plt)})
    plt.close()

    # ---------- artifact ----------
    torch.save(model.state_dict(), "model.pth")
    artifact = wandb.Artifact(
        name="corrupt_mnist_model",
        type="model",
        metadata=final_metrics,
    )
    artifact.add_file("model.pth")
    run.log_artifact(artifact)

    run.finish()
    logger.info("Training completed")


def main():
    typer.run(train)


if __name__ == "__main__":
    main()