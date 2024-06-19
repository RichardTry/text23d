import asyncio

# import torch.cuda
import torch
from rich.console import Console
from torch.utils.data import DataLoader

from camera_view_generator.camera import CameraDataset
from stable_diffusion_guidance.guidance import StableDiffusionGuidance
from tiny_nerf.nerf import NerfModel, train

# from PIL import Image


console = Console()


async def run(prompt: str):
    console.print(
        ":film_projector: [bold green] Welcome to Tiny DreamFusion :film_projector:"
    )
    print(f"Training {prompt}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    height = width = 32
    # The smaller the images the faster the training and evaluation
    batch_size = 8  # If running out of memory reduce this
    max_frames = (
        5  # Maximum number of images to train with. Lower this to speed up training
    )
    nb_epochs = 1
    console.print(f"Using {max_frames} images and running for {nb_epochs} epochs")

    training_dataset = CameraDataset(label="train", samples=128, H=height, W=width)
    testing_dataset = CameraDataset(label="test", samples=128, H=height, W=width)


    model = NerfModel(hidden_dim=256).to(device)
    model_optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        model_optimizer, milestones=[2, 4, 8], gamma=0.5
    )
    data_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
    sd_guidance = StableDiffusionGuidance(prompt)
    train(
        batch_size,
        model,
        model_optimizer,
        scheduler,
        data_loader,
        testing_dataset,
        guidance=sd_guidance,
        nb_epochs=nb_epochs,
        device=device,
        hn=2,
        hf=6,
        nb_bins=192,
        H=height,
        W=width,
    )

if __name__ == "__main__":
    asyncio.run(run('Delicious hamburger'))