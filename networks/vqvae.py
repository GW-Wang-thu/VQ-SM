import torch.nn as nn
import torch.nn.functional as F

import torch
from torch.autograd import Function


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1 / self._num_embeddings, 1 / self._num_embeddings)
        self._commitment_cost = commitment_cost

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings


class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost

        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()

        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * torch.sum(encodings, 0)

            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                    (self._ema_cluster_size + self._epsilon)
                    / (n + self._num_embeddings * self._epsilon) * n)

            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)

            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings


class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 1),
            nn.BatchNorm2d(dim)
        )
    def forward(self, x):
        return x + self.block(x)


class ResBlockL(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.BatchNorm2d(dim)
        )
    def forward(self, x):
        return x + self.block(x)


class VQVAE_Simp(nn.Module):
    def __init__(self, opt, dim=64, embedding_dim=64, num_embeddings=2048, beta=0.25, decay=0):
        super().__init__()
        if opt == 'SDEG':
            self.input_dim = 15
        else:
            self.input_dim = 16
        if decay > 0:
            self._vq_vae = VectorQuantizerEMA(num_embeddings, embedding_dim,
                                              beta, decay)
        else:
            self._vq_vae = VectorQuantizer(num_embeddings, embedding_dim,
                                           beta)
        self.data_variance = 1.0
        self.Encoder = nn.Sequential(
            nn.Conv2d(in_channels=self.input_dim, out_channels=dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(dim),
            ResBlock(dim),
            ResBlock(dim),
            ResBlock(dim),
            ResBlock(dim),
            nn.Conv2d(dim, embedding_dim, kernel_size=1, stride=1, padding=0),
        )

        self.Decoder = nn.Sequential(
            nn.Conv2d(embedding_dim, dim, kernel_size=3, stride=1, padding=1),
            ResBlock(dim),
            ResBlock(dim),
            ResBlock(dim),
            ResBlock(dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, dim, 4, 2, 1),
            # nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, dim, 4, 2, 1),
            # nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, self.input_dim, 4, 2, 1),
            nn.Tanh()
        )
        # self.apply(weights_init)

    def encode(self, x):
        z_e_x = self.Encoder(x)
        loss, quantized, perplexity, encodings = self._vq_vae(z_e_x)
        return loss, quantized, perplexity, encodings      # quantized 是 one-hot 编码向量

    def decode(self, latents):
        x_recon = self.Decoder(latents)
        return x_recon

    def training_step(self, x):
        loss_vq, quantized, perplexity, encodings = self.encode(x)
        x_recon = self.decode(quantized)
        loss_recons = F.mse_loss(x_recon, x) / self.data_variance
        loss = loss_recons + loss_vq
        return x_recon, loss, loss_recons.item(), loss_vq.item()

    def validation_step(self, x):
        loss_vq, quantized, perplexity, encodings = self.encode(x)
        x_recon = self.decode(quantized)
        loss_recons = F.mse_loss(x_recon, x) / self.data_variance
        loss = loss_recons + loss_vq
        return x_recon, loss.item(), loss_recons.item(), loss_vq.item()


class Predictor(nn.Module):
    def __init__(self, embedding_dim=64, size=12):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.size = size
        self.predictor = nn.Sequential(
            nn.Linear(3, embedding_dim),
            nn.Tanh(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.Tanh(),
            nn.Linear(embedding_dim, embedding_dim * size * size),
        )

    def forward(self, x):
        b, v = x.size()
        recon_vect = self.predictor(x)
        latents = torch.reshape(recon_vect, (b, self.embedding_dim, self.size, self.size))
        return latents


class simple_CNN(nn.Module):
    def __init__(self, opt, embedding_dim=64, size=12, dim=64):
        super().__init__()

        if opt == 'SDEG':
            self.input_dim = 15
        else:
            self.input_dim = 16
        self.embedding_dim = embedding_dim
        self.size = size

        self.predictor = nn.Sequential(
            nn.Linear(3, embedding_dim),
            nn.Tanh(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.Tanh(),
            nn.Linear(embedding_dim, embedding_dim * size * size),
        )

        self.res_block = ResBlock(dim=embedding_dim)

        self.decoder = nn.Sequential(
            nn.Conv2d(embedding_dim, dim, kernel_size=3, stride=1, padding=1),
            ResBlock(dim),
            ResBlock(dim),
            ResBlock(dim),
            ResBlock(dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, dim, 4, 2, 1),
            # nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, dim, 4, 2, 1),
            # nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, self.input_dim, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        b, v = x.size()
        recon_vect = self.predictor(x)
        latents = torch.reshape(recon_vect, (b, self.embedding_dim, self.size, self.size))
        pre = self.decoder(latents)
        return pre


class simple_NN(nn.Module):
    def __init__(self, opt, embedding_dim=64, size=12, dim=64):
        super().__init__()

        if opt == 'SDEG':
            self.input_dim = 15
        else:
            self.input_dim = 16
        self.embedding_dim = embedding_dim
        self.size = size

        self.predictor = nn.Sequential(
            nn.Linear(3, embedding_dim),
            nn.Tanh(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.Tanh(),
            nn.Linear(embedding_dim, embedding_dim * size * size),
        )

        self.res_block = ResBlockL(dim=dim * size * size)

        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim * size * size, dim * size * size),
            ResBlock(dim * size * size),
            ResBlock(dim * size * size),
            ResBlock(dim * size * size),
            ResBlock(dim * size * size),
            nn.ReLU(True),
            nn.Linear(dim * size * size, dim * size * size * 4),
            nn.ReLU(True),
            nn.Linear(dim * size * size, dim * size * size * 4),
            nn.ReLU(True),
            nn.Linear(dim * size * size, self.input_dim * size * size),
            nn.Tanh()
        )

    def forward(self, x):
        b, v = x.size()
        recon_vect = self.predictor(x)
        pre = self.decoder(recon_vect)
        pre_imgs = torch.reshape(pre, (b, self.input_dim, self.size, self.size))
        return pre_imgs
