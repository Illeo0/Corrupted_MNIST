from corrupted_mnist.data import corrupt_mnist
from corrupted_mnist.model import MyModel

import torch
import typer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

app = typer.Typer()

@app.command()
def evaluate(model_checkpoint: str = "models/model.pth", batch_size: int = 32) -> None:
    """Evaluate a trained model on the corrupted MNIST test set."""
    print("Evaluation started...")
    print(f"Checkpoint: {model_checkpoint}")

    model = MyModel().to(DEVICE)
    state_dict = torch.load(model_checkpoint, map_location=DEVICE)
    model.load_state_dict(state_dict)

    _, test_dataset = corrupt_mnist()
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    print(f"Test accuracy: {correct / total:.4f}")

if __name__ == "__main__":
    typer.run(evaluate)