import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


def gan_train(model, ds, optimizer, criterion, batch_size, **kwargs):
    """Train a :class:`~stocknet.nets.gan.TimeGAN` for one epoch.

    Alternates between training the discriminator on real/fake pairs and
    training the generator to fool the discriminator.

    Args:
        model:      :class:`~stocknet.nets.gan.TimeGAN` instance
        ds:         dataset returning ``(src, tgt, ...)`` batches
        optimizer:  generator optimizer (created from the training config)
        criterion:  adversarial loss; defaults to ``BCELoss`` when ``None``
        batch_size: number of samples per mini-batch

    Returns:
        float: mean combined (generator + discriminator) loss for the epoch
    """
    model.train()
    ds.train()

    # Lazy creation of the discriminator optimizer with the same lr as the generator
    if not hasattr(model, "_d_optimizer"):
        lr = optimizer.param_groups[0]["lr"]
        model._d_optimizer = torch.optim.Adam(model.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    adversarial_loss = criterion if criterion is not None else nn.BCELoss()
    device = model.device
    d_losses, g_losses = [], []

    end_index = len(ds)
    for index in tqdm(range(0, end_index - batch_size, batch_size)):
        batch = ds[index : index + batch_size]
        src, tgt = batch[0].to(device), batch[1].to(device)
        bs = src.size(0)

        real_labels = torch.ones(bs, 1, device=device)
        fake_labels = torch.zeros(bs, 1, device=device)

        # --- Discriminator step ---
        model._d_optimizer.zero_grad()

        real_pred = model.discriminator(src, tgt)
        d_real_loss = adversarial_loss(real_pred, real_labels)

        noise = torch.randn(bs, model.latent_dim, device=device)
        fake_tgt = model.generator(src, noise).detach()
        fake_pred = model.discriminator(src, fake_tgt)
        d_fake_loss = adversarial_loss(fake_pred, fake_labels)

        d_loss = (d_real_loss + d_fake_loss) / 2
        d_loss.backward()
        model._d_optimizer.step()
        d_losses.append(d_loss.item())

        # --- Generator step ---
        optimizer.zero_grad()

        noise = torch.randn(bs, model.latent_dim, device=device)
        fake_tgt = model.generator(src, noise)
        fake_pred = model.discriminator(src, fake_tgt)
        g_loss = adversarial_loss(fake_pred, real_labels)
        g_loss.backward()
        optimizer.step()
        g_losses.append(g_loss.item())

    return float(np.mean(g_losses) + np.mean(d_losses))


def gan_eval(model, ds, criterion, batch_size, **kwargs):
    """Evaluate a :class:`~stocknet.nets.gan.TimeGAN`.

    Measures the generator's ability to fool the discriminator on held-out data.
    A loss near ``log(2) ≈ 0.693`` (random chance) indicates the generator is
    producing convincing sequences.

    Args:
        model:      :class:`~stocknet.nets.gan.TimeGAN` instance
        ds:         dataset in evaluation mode
        criterion:  adversarial loss; defaults to ``BCELoss`` when ``None``
        batch_size: number of samples per mini-batch

    Returns:
        float: mean generator fooling loss on the validation set
    """
    model.eval()
    ds.eval()

    adversarial_loss = criterion if criterion is not None else nn.BCELoss()
    device = model.device
    g_losses = []

    end_index = len(ds)
    with torch.no_grad():
        for index in tqdm(range(0, end_index - batch_size, batch_size)):
            batch = ds[index : index + batch_size]
            src = batch[0].to(device)
            bs = src.size(0)

            real_labels = torch.ones(bs, 1, device=device)
            noise = torch.randn(bs, model.latent_dim, device=device)
            fake_tgt = model.generator(src, noise)
            fake_pred = model.discriminator(src, fake_tgt)
            g_loss = adversarial_loss(fake_pred, real_labels)
            g_losses.append(g_loss.item())

    return float(np.mean(g_losses))
