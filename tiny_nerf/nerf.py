import os

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from rich.console import Console
from rich.progress import Progress

console = Console()


def _take(chunk_size, rays):
    batch_idxs = np.arange(0, rays.shape[0], chunk_size)
    for batch_start in batch_idxs:
        batch_end = batch_start + chunk_size
        batch_end = rays.shape[0] if batch_end > rays.shape[0] else batch_end
        yield rays[batch_start:batch_end, :]


@torch.no_grad()
def test(model, hn, hf, dataset, chunk_size=10, img_index=0, nb_bins=192, H=400, W=400):
    """
    Args:
        hn: near plane distance
        hf: far plane distance
        dataset: dataset to render
        chunk_size (int, optional): chunk size for memory efficiency. Defaults to 10.
        img_index (int, optional): image index to render. Defaults to 0.
        nb_bins (int, optional): number of bins for density estimation. Defaults to 192.
        H (int, optional): image height. Defaults to 400.
        W (int, optional): image width. Defaults to 400.

    Returns:
        None: None
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs("novel_views", exist_ok=True)
    data = []  # list of regenerated pixel values
    example = dataset[img_index]
    for chunk in _take(chunk_size, example["rays"]):
        # iterate over chunks
        ray_origins = chunk[:, :3].to(device)
        ray_directions = chunk[:, 3:6].to(device)

        regenerated_px_values = render_rays(
            model, ray_origins, ray_directions, hn=hn, hf=hf, nb_bins=nb_bins
        )
        data.append(regenerated_px_values)
    img = torch.cat(data).data.cpu().numpy().reshape(H, W, 3)

    pil_img = Image.fromarray((img * 255).astype(np.uint8))
    pil_img.save(f"novel_views/img_{img_index}.png")


class NerfModel(nn.Module):
    def __init__(self, embedding_dim_pos=10, embedding_dim_direction=4, hidden_dim=128):
        super(NerfModel, self).__init__()

        self.block1 = nn.Sequential(
            nn.Linear(embedding_dim_pos * 6 + 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        # density estimation
        self.block2 = nn.Sequential(
            nn.Linear(embedding_dim_pos * 6 + hidden_dim + 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim + 1),
        )
        # color estimation
        self.block3 = nn.Sequential(
            nn.Linear(embedding_dim_direction * 6 + hidden_dim + 3, hidden_dim // 2),
            nn.ReLU(),
        )
        self.block4 = nn.Sequential(
            nn.Linear(hidden_dim // 2, 3),
            nn.Sigmoid(),
        )

        self.embedding_dim_pos = embedding_dim_pos
        self.embedding_dim_direction = embedding_dim_direction
        self.relu = nn.ReLU()

    @staticmethod
    def positional_encoding(x, L):
        out = [x]
        for j in range(L):
            out.append(torch.sin(2**j * x))
            out.append(torch.cos(2**j * x))
        return torch.cat(out, dim=1)

    def forward(self, o, d):
        emb_x = self.positional_encoding(
            o, self.embedding_dim_pos
        )  # emb_x: [batch_size, embedding_dim_pos * 6]
        emb_d = self.positional_encoding(
            d, self.embedding_dim_direction
        )  # emb_d: [batch_size, embedding_dim_direction * 6]
        h = self.block1(emb_x)  # h: [batch_size, hidden_dim]
        tmp = self.block2(
            torch.cat((h, emb_x), dim=1)
        )  # tmp: [batch_size, hidden_dim + 1]
        h, sigma = tmp[:, :-1], self.relu(
            tmp[:, -1]
        )  # h: [batch_size, hidden_dim], sigma: [batch_size]
        h = self.block3(torch.cat((h, emb_d), dim=1))  # h: [batch_size, hidden_dim // 2]
        c = self.block4(h)  # c: [batch_size, 3]
        return c, sigma


def compute_accumulated_transmittance(alphas):
    accumulated_transmittance = torch.cumprod(alphas, 1)
    return torch.cat(
        (
            torch.ones((accumulated_transmittance.shape[0], 1), device=alphas.device),
            accumulated_transmittance[:, :-1],
        ),
        dim=-1,
    )


def render_rays(nerf_model, ray_origins, ray_directions, hn=0, hf=0.5, nb_bins=192):
    device = ray_origins.device

    # print(ray_origins.shape)
    # print('squizeeng...')

    ray_origins = ray_origins.view(-1, ray_origins.size(-1))

    # print(ray_origins.shape)
    # print(ray_directions.shape)
    # print('squizeeng...')
    ray_directions = ray_directions.view(-1, ray_directions.size(-1))
    # print(ray_directions.shape)

    t = torch.linspace(hn, hf, nb_bins, device=device).expand(
        ray_origins.shape[0], nb_bins
    )
    # print(t.shape)

    # Perturb sampling along each ray.
    mid = (t[:, :-1] + t[:, 1:]) / 2.0
    lower = torch.cat((t[:, :1], mid), -1)
    upper = torch.cat((mid, t[:, -1:]), -1)
    u = torch.rand(t.shape, device=device)
    t = lower + (upper - lower) * u  # [batch_size, nb_bins]
    # print('>>>>', t.shape)
    delta = torch.cat(
        (
            t[:, 1:] - t[:, :-1],
            torch.tensor([1e10], device=device).expand(ray_origins.shape[0], 1),
        ),
        -1,
    )

    # Compute the 3D points along each ray
    # print(ray_directions.unsqueeze(1).shape)
    # print(t.unsqueeze(2).shape)
    vecs = t.unsqueeze(2) * ray_directions.unsqueeze(1)
    x = ray_origins.unsqueeze(1) + vecs  # [batch_size, nb_bins, 3]
    # Expand the ray_directions tensor to match the shape of x
    ray_directions = ray_directions.expand(nb_bins, ray_directions.shape[0], 3).transpose(
        0, 1
    )
    colors, sigma = nerf_model(x.reshape(-1, 3), ray_directions.reshape(-1, 3))
    colors = colors.reshape(x.shape)
    sigma = sigma.reshape(x.shape[:-1])
    alpha = 1 - torch.exp(-sigma * delta)  # [batch_size, nb_bins]
    weights = compute_accumulated_transmittance(1 - alpha).unsqueeze(2) * alpha.unsqueeze(
        2
    )
    # Compute the pixel values as a weighted sum of colors along each ray
    c = (weights * colors).sum(dim=1).to(device)
    weight_sum = weights.sum(-1).sum(-1)  # Regularization for white background
    return c + 1 - weight_sum.unsqueeze(-1).to(device)


def train(
    batch_size,
    nerf_model,
    optimizer,
    scheduler,
    data_loader,
    testing_dataset,
    guidance,
    device="cuda",
    hn=0,
    hf=1,
    nb_epochs=int(1e5),
    nb_bins=192,
    H=400,
    W=400,
):
    model_save_path = "model_saved.pt"
    training_loss = []

    if os.path.exists(model_save_path):
        nerf_model.load_state_dict(torch.load(model_save_path))
    else:
        with Progress() as progress:
            epochs_task = progress.add_task("[red]Epoch...", total=nb_epochs)

            for epoch_idx in range(nb_epochs):
                progress.update(
                    epochs_task, description=f"[red]Epoch {epoch_idx}/{nb_epochs}"
                )
                batch_task = progress.add_task("[green]Batch", total=len(data_loader))

                for idx, batch in enumerate(data_loader):
                    #print('rays shape', batch['rays'].shape)
                    ray_origins = batch["rays"][:, :, :3].to(device)
                    #print('origins', ray_origins.shape)
                    ray_directions = batch["rays"][:, :, 3:6].to(device)
                    #print('directions', ray_directions.shape)
                    concrete_directions = batch["dirs"].to(device)

                    regenerated_px_values = render_rays(
                        nerf_model,
                        ray_origins,
                        ray_directions,
                        hn=hn,
                        hf=hf,
                        nb_bins=nb_bins,
                    )

                    # loss = ((ground_truth_px_values - regenerated_px_values) ** 2).sum()
                    loss = guidance.calculate_loss(
                        regenerated_px_values, batch_size, H, W, concrete_directions
                    )
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    training_loss.append(loss.item())
                    progress.update(
                        batch_task,
                        advance=1,
                        description=f"[green]Batch {idx}/{len(data_loader)}"
                        f"loss: {np.mean(training_loss)}",
                    )
                scheduler.step()
                progress.update(epochs_task, advance=1)
            torch.save(nerf_model.state_dict(), model_save_path)

    for img_index in range(len(testing_dataset)):
        test(nerf_model, hn, hf, testing_dataset, chunk_size=1024,
        img_index=img_index, nb_bins=nb_bins, H=H, W=W)
    return training_loss
