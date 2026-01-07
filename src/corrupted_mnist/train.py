import matplotlib.pyplot as plt
import torch
import typer
from corrupted_mnist.data import corrupt_mnist
from corrupted_mnist.model import MyModel
from pathlib import Path



DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

app = typer.Typer()

@app.command()
def train(lr: float = 1e-3, batch_size: int = 32, epochs: int = 10) -> None:
    print("Training started...")
    print(f"{lr=}, {batch_size=}, {epochs=}")

    model = MyModel().to(DEVICE)
    train_dataset, _ = corrupt_mnist()

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    statistics = {"train_loss": [], "train_accuracy": []}
    for epoch in range(epochs):
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            statistics["train_loss"].append(loss.item())

            accuracy = (outputs.argmax(dim=1) == labels).float().mean().item()
            statistics["train_accuracy"].append(accuracy)

            if i % 100 == 0:
                print(f"Epoch {epoch}, iter {i}, loss: {loss.item()}")
            
    
    print("Training finished.")
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(statistics["train_loss"])
    axs[0].set_title("Train loss")
    axs[1].plot(statistics["train_accuracy"])
    axs[1].set_title("Train accuracy")
   
    Path("models").mkdir(parents=True, exist_ok=True)
    Path("reports/figures").mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), "models/model.pth")
    fig.savefig("reports/figures/training_statistics.png")
    plt.close(fig)
if __name__ == "__main__":
    typer.run(train)
