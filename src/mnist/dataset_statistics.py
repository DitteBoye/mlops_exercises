import matplotlib.pyplot as plt
import torch
import typer

from mnist.dataset import MnistDataset


def dataset_statistics(datadir: str = "data") -> None:
    train_dataset = MnistDataset(data_folder=datadir, train=True)
    test_dataset = MnistDataset(data_folder=datadir, train=False)

    print(f"Train dataset: {train_dataset.name}")
    print(f"Number of images: {len(train_dataset)}")
    print(f"Image shape: {train_dataset[0][0].shape}")
    print()
    print(f"Test dataset: {test_dataset.name}")
    print(f"Number of images: {len(test_dataset)}")
    print(f"Image shape: {test_dataset[0][0].shape}")

    # ---- example images ----
    images = train_dataset.images[:25]
    fig, axes = plt.subplots(5, 5, figsize=(5, 5))
    for ax, img in zip(axes.flatten(), images):
        ax.imshow(img.squeeze(), cmap="gray")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("mnist_images.png")
    plt.close()

    # ---- label distributions ----
    train_dist = torch.bincount(train_dataset.target)
    test_dist = torch.bincount(test_dataset.target)

    plt.bar(range(10), train_dist)
    plt.title("Train label distribution")
    plt.savefig("train_label_distribution.png")
    plt.close()

    plt.bar(range(10), test_dist)
    plt.title("Test label distribution")
    plt.savefig("test_label_distribution.png")
    plt.close()


if __name__ == "__main__":
    typer.run(dataset_statistics)