import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler, StableDiffusionPipeline


class StableDiffusionGuidance:
    def __init__(self, prompt):
        self.model_key = "stabilityai/stable-diffusion-2-1-base"
        self.torch_device = "cpu"
        pipe = StableDiffusionPipeline.from_pretrained(
            self.model_key, torch_dtype=torch.float32
        )
        pipe.to(self.torch_device)
        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet
        self.scheduler = DDIMScheduler.from_pretrained(
            self.model_key, subfolder="scheduler", torch_dtype=torch.float32
        )
        # from paper
        self.t_range = [0.02, 0.98]
        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * self.t_range[0])
        self.max_step = int(self.num_train_timesteps * self.t_range[1])
        self.text_embeddings = {}
        self.directions = ["front", "side", "back", "bottom", "overhead"]
        self.alphas = self.scheduler.alphas_cumprod.to(self.torch_device)
        self.create_text_embedding_dict(prompt)

    def calculate_loss(
        self,
        generated_render,
        batch_size,
        H,
        W,
        concrete_directions,
        guidance_scale=100,
        grad_scale=1,
    ):
        generated_render = generated_render.reshape(batch_size, H, W, 3).permute(
            0, 3, 1, 2
        )
        generated_render_i512 = F.interpolate(
            generated_render, (512, 512), mode="bilinear", align_corners=False
        )

        # encode image into latents with vae, requires grad!
        posterior = self.vae.encode(generated_render_i512).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(
            self.min_step,
            self.max_step + 1,
            (latents.shape[0],),
            dtype=torch.long,
            device=self.torch_device,
        )

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # create noise
            noise = torch.randn_like(latents)
            # add noise to the latent representation
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # twice as much latents in order to run both
            # guided generation and unconditional generation
            latent_model_input = torch.cat([latents_noisy] * 2)
            tt = torch.cat([t] * 2)
            # map directions to the text prompt
            text_embeddings = self.map_directions_to_text_embeddings(concrete_directions)
            noise_pred = self.unet(
                latent_model_input, tt, encoder_hidden_states=text_embeddings
            ).sample

            # perform guidance, guidance in paper was around 100
            noise_pred_uncond, noise_pred_pos = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_pos - noise_pred_uncond
            )

        # calculate SDS
        grad_scale = 1
        w = 1 - self.alphas[t]

        grad = grad_scale * w[:, None, None, None] * (noise_pred - noise)
        grad = torch.nan_to_num(grad)

        targets = (latents - grad).detach()
        loss = (
            0.5 * F.mse_loss(latents.float(), targets, reduction="sum") / latents.shape[0]
        )
        return loss

    def create_text_embedding_dict(self, prompt):
        for direction in self.directions:
            extended_prompt = f"{prompt} {direction} view"
            text_input = self.tokenizer(
                extended_prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            with torch.no_grad():
                self.text_embeddings[direction] = self.text_encoder(
                    text_input.input_ids.to(self.torch_device)
                )[0]

        max_length = text_input.input_ids.shape[-1]
        uncond_input = self.tokenizer(
            [""], padding="max_length", max_length=max_length, return_tensors="pt"
        )
        with torch.no_grad():
            self.text_embeddings["unconditional"] = self.text_encoder(
                uncond_input.input_ids.to(self.torch_device)
            )[0]

    def map_directions_to_text_embeddings(self, directions):
        cond_embeddings = [self.text_embeddings[str(dir)] for dir in directions]
        cond_embeddings = torch.cat(cond_embeddings, 0)
        uncond_embeddings = [
            self.text_embeddings["unconditional"] for _ in range(len(directions))
        ]
        uncond_embeddings = torch.cat(uncond_embeddings, 0)
        return torch.cat([cond_embeddings, uncond_embeddings], 0)
