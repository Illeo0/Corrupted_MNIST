from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch
import typer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from corrupted_mnist.model import MyModel  # adjust if your class name differs
from corrupted_mnist.data import corrupt_mnist  # uses processed data

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)

app = typer.Typer()


def _get_feature_extractor(model: torch.nn.Module) -> torch.nn.Module:
    """Replace the classifier head with identity to expose penultimate features."""
    candidate_heads = ("fc", "fc1", "classifier")
    for head_name in candidate_heads:
        if hasattr(model, head_name):
            setattr(model, head_name, torch.nn.Identity())
            return model
    raise AttributeError(
        "Could not locate a final layer named any of: "
        f"{', '.join(candidate_heads)} on MyModel. Update _get_feature_extractor()."
    )


@app.command()
def visualize(
    model_checkpoint: str = "models/model.pth",
    figure_name: str = "tsne_embeddings.png",
    split: str = "train",
    max_points: int = 5000,
    batch_size: int = 128,
    pca_dim: int = 100,
    seed: int = 42,
) -> None:
    """
    Create a t-SNE plot of feature embeddings from a pretrained model and save to reports/figures/.
    """
    # Ensure output directory exists
    out_dir = Path("reports/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model: torch.nn.Module = MyModel().to(DEVICE)
    state_dict = torch.load(model_checkpoint, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    # Convert model into feature extractor
    model = _get_feature_extractor(model)

    # Load data (processed via corrupt_mnist)
    train_set, test_set = corrupt_mnist()
    dataset = train_set if split.lower() == "train" else test_set

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    embeddings_list = []
    targets_list = []

    with torch.inference_mode():
        seen = 0
        for images, targets in loader:
            images = images.to(DEVICE)
            feats = model(images)  # features before final classifier
            embeddings_list.append(feats.detach().cpu())
            targets_list.append(targets.detach().cpu())

            seen += images.size(0)
            if seen >= max_points:
                break

    embeddings = torch.cat(embeddings_list, dim=0).numpy()
    targets = torch.cat(targets_list, dim=0).numpy()

    # Optional PCA pre-reduction (recommended before t-SNE for high-dimensional features)
    if embeddings.shape[1] > pca_dim:
        embeddings = PCA(n_components=pca_dim, random_state=seed).fit_transform(embeddings)

    # t-SNE to 2D
    tsne = TSNE(n_components=2, random_state=seed, init="pca", learning_rate="auto")
    emb_2d = tsne.fit_transform(embeddings)

    # Plot
    plt.figure(figsize=(10, 10))
    for cls in sorted(set(targets.tolist())):
        mask = targets == cls
        plt.scatter(emb_2d[mask, 0], emb_2d[mask, 1], label=str(cls), s=8)

    plt.legend(title="Digit", markerscale=2)
    plt.title(f"t-SNE of {split} features (n={len(targets)})")
    out_path = out_dir / figure_name
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved figure to: {out_path}")


if __name__ == "__main__":
    typer.run(visualize)
