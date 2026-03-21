import torch
import torch.nn as nn


class TimeGANGenerator(nn.Module):
    """LSTM-based conditional generator for financial time series.

    Takes a source (conditioning) sequence plus latent noise and generates
    a synthetic target sequence of the requested length.

    Args:
        input_dim:   feature dimension of the source sequence
        latent_dim:  size of the noise vector
        hidden_dim:  LSTM hidden size (used in both encoder and decoder)
        output_len:  number of time steps to generate
        output_dim:  feature dimension of the generated sequence
        device:      torch device
        num_layers:  number of stacked LSTM layers (default: 2)
    """

    def __init__(self, input_dim, latent_dim, hidden_dim, output_len, output_dim, device, num_layers=2, **kwargs):
        self.args = {
            "input_dim": input_dim,
            "latent_dim": latent_dim,
            "hidden_dim": hidden_dim,
            "output_len": output_len,
            "output_dim": output_dim,
            "num_layers": num_layers,
        }
        super().__init__()
        self.output_len = output_len
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.device = device

        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.noise_proj = nn.Linear(latent_dim, hidden_dim * num_layers)
        self.decoder = nn.LSTM(output_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, src, noise=None):
        src = src.to(self.device)
        batch_size = src.size(0)

        _, (hidden, cell) = self.encoder(src)

        if noise is not None:
            noise = noise.to(self.device)
            noise_h = self.noise_proj(noise)  # (batch, hidden_dim * num_layers)
            noise_h = noise_h.view(batch_size, self.num_layers, self.hidden_dim).transpose(0, 1).contiguous()
            hidden = hidden + noise_h

        # Autoregressive decoding
        decoder_input = torch.zeros(batch_size, 1, self.output_dim, device=self.device)
        outputs = []
        h, c = hidden, cell
        for _ in range(self.output_len):
            out, (h, c) = self.decoder(decoder_input, (h, c))
            step = self.output_proj(out)  # (batch, 1, output_dim)
            outputs.append(step)
            decoder_input = step

        return torch.cat(outputs, dim=1)  # (batch, output_len, output_dim)


class TimeGANDiscriminator(nn.Module):
    """LSTM-based discriminator for financial time series.

    Receives the conditioning source sequence concatenated with a target
    sequence (real or generated) and outputs a real/fake probability.

    Args:
        input_dim:  feature dimension shared by source and target
        hidden_dim: LSTM hidden size
        device:     torch device
        num_layers: number of stacked LSTM layers (default: 2)
    """

    def __init__(self, input_dim, hidden_dim, device, num_layers=2, **kwargs):
        self.args = {"input_dim": input_dim, "hidden_dim": hidden_dim, "num_layers": num_layers}
        super().__init__()
        self.device = device

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, src, tgt):
        src = src.to(self.device)
        tgt = tgt.to(self.device)
        x = torch.cat([src, tgt], dim=1)  # (batch, src_len + tgt_len, input_dim)
        _, (hidden, _) = self.lstm(x)
        return self.classifier(hidden[-1])  # (batch, 1)


class TimeGAN(nn.Module):
    """Conditional Time-series GAN for financial sequence generation.

    Wraps a :class:`TimeGANGenerator` and a :class:`TimeGANDiscriminator`.
    The external optimizer (loaded from the training config) is applied to the
    *generator* only; the discriminator maintains its own optimizer which is
    created lazily inside the GAN trainer on the first training step.

    Config example::

        {
          "key": "TimeGAN",
          "model_name": "gan_forex",
          "params": {
            "input_dim": 5,
            "latent_dim": 32,
            "hidden_dim": 128,
            "output_len": 10,
            "output_dim": 5
          }
        }

    Training config should use ``"BCELoss"`` as the loss function and specify
    ``"trainer": {"train_key": "gantrainer.gan_train",
    "eval_key": "gantrainer.gan_eval"}`` or rely on the automatic GAN trainer
    selection built into :mod:`stocknet.trainer.factory`.

    Args:
        input_dim:   feature dimension of source/target sequences
        latent_dim:  noise vector size fed to the generator
        hidden_dim:  LSTM hidden size for both G and D
        output_len:  number of time steps the generator produces
        output_dim:  feature dimension of generated sequences
        device:      torch device
        num_layers:  stacked LSTM layers (default: 2)
    """

    def __init__(self, input_dim, latent_dim, hidden_dim, output_len, output_dim, device, num_layers=2, **kwargs):
        self.args = {
            "input_dim": input_dim,
            "latent_dim": latent_dim,
            "hidden_dim": hidden_dim,
            "output_len": output_len,
            "output_dim": output_dim,
            "num_layers": num_layers,
        }
        super().__init__()
        self.latent_dim = latent_dim
        self.device = device

        self.generator = TimeGANGenerator(input_dim, latent_dim, hidden_dim, output_len, output_dim, device, num_layers)
        self.discriminator = TimeGANDiscriminator(input_dim, hidden_dim, device, num_layers)

    def forward(self, src, noise=None):
        if noise is None:
            noise = torch.randn(src.size(0), self.latent_dim, device=self.device)
        return self.generator(src, noise)

    def parameters(self, recurse=True):
        """Expose only generator parameters for the external (config-driven) optimizer."""
        return self.generator.parameters(recurse)

    @classmethod
    def load(cls, **kwargs):
        return cls(**kwargs)
