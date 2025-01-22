import typer
from utils import save_model, plot_train_val_losses

app = typer.Typer()

@app.command()
def train(
    image_path: str = typer.Option(..., help="Path to the dataset images."),
    model_name: str = typer.Option("resnet50", help="Model architecture name."),
    num_classes: int = typer.Option(..., help="Number of output classes."),
    batch_size: int = typer.Option(16, help="Batch size for training and validation."),
    learning_rate: float = typer.Option(1e-3, help="Learning rate for the optimizer."),
    weight_decay: float = typer.Option(1e-4, help="Weight decay for the optimizer."),
    num_epochs: int = typer.Option(10, help="Number of epochs for training."),
    save_path: str = typer.Option("../model.pth", help="Path to save the trained model."),
    profile: bool = typer.Option(False, help="Enable PyTorch Profiler."),
):
    """
    Train the model using the given dataset and parameters.
    Optionally enable profiling with the --profile parameter.
    """
    from datasets import CustomDataset
    from torch.utils.data import DataLoader, random_split
    from torch.optim import Adam
    from torch import nn
    import torch
    from model import get_model
    from train import train
    from torch.profiler import profile as torch_profile, ProfilerActivity
    from torch.utils.tensorboard import SummaryWriter

    # Dataset setup
    dataset = CustomDataset(image_path, model_name, num_classes)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Model setup
    model = get_model(model_name, num_classes, pretrained=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    def train_with_or_without_profiling(model, train_dataloader, val_dataloader, criterion, optimizer, device, num_epochs, profile=False):
        # Set up TensorBoard writer if profiling is enabled
        writer = SummaryWriter('./log') if profile else None

        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)

        if profile:
            with torch_profile(
                activities=activities,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(f'./log/{model_name}'),
                record_shapes=True,
                with_stack=True
            ) as prof:
                # Call the regular training loop
                train(model, train_dataloader, val_dataloader, criterion, optimizer, device, num_epochs, prof)

                # Optionally export the trace to a Chrome trace JSON file
                prof.export_chrome_trace("trace.json")

                # Print the profiling table
                if torch.cuda.is_available():
                    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
                else:
                    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

        else:
            # Call the regular training loop without profiling
            train(model, train_dataloader, val_dataloader, criterion, optimizer, device, num_epochs)

        if writer:
            writer.close()

    # Call the training function with profiling flag from the CLI
    train_with_or_without_profiling(model, train_dataloader, val_dataloader, criterion, optimizer, device, num_epochs, profile)

    # Save the trained model
    save_model(model, save_path)
    typer.echo(f"Model saved to {save_path}")


@app.command()
def test_model(
    model_path: str = typer.Option(..., help="Path to the trained model."),
    image_path: str = typer.Option(..., help="Path to the test dataset."),
    num_classes: int = typer.Option(..., help="Number of output classes."),
    batch_size: int = typer.Option(16, help="Batch size for testing."),
):
    """
    Test the model using the specified dataset and model file.
    """
    from torch.utils.data import DataLoader
    from model import get_model
    from test import test
    from datasets import CustomDataset
    import torch

    dataset = CustomDataset(image_path, "resnet50", num_classes)
    test_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model("resnet50", num_classes, pretrained=False)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)

    # Test
    test(model, test_dataloader, device)


if __name__ == "__main__":
    app()
