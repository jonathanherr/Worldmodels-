"""v0 world model: conv autoencoder + deterministic GRU dynamics in latent space.

    z_t   = encoder(x_t)                      # what the frame shows
    h_t+1 = GRU(h_t, [z_t, onehot(a_t)])      # memory: integrates motion over time
    z^t+1 = prior(h_t+1)                      # predicted next latent
    x^    = decoder(z), r^ = reward_head(h), c^ = continue_head(h)

During an imagined rollout the model feeds its own prediction z^ back in place of the
encoder's z, so no real frames are needed after the context window. v1 will replace this
deterministic latent with Dreamer's stochastic RSSM.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .env import NUM_ACTIONS


class Encoder(nn.Module):
    def __init__(self, latent_dim: int, ch: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, ch, 4, 2, 1), nn.SiLU(),            # 32
            nn.Conv2d(ch, ch * 2, 4, 2, 1), nn.SiLU(),       # 16
            nn.Conv2d(ch * 2, ch * 4, 4, 2, 1), nn.SiLU(),   # 8
            nn.Conv2d(ch * 4, ch * 4, 4, 2, 1), nn.SiLU(),   # 4
            nn.Flatten(),
            nn.Linear(ch * 4 * 16, latent_dim),
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim: int, ch: int = 32):
        super().__init__()
        self.ch = ch
        self.fc = nn.Linear(latent_dim, ch * 4 * 16)
        self.net = nn.Sequential(
            nn.SiLU(),
            nn.ConvTranspose2d(ch * 4, ch * 4, 4, 2, 1), nn.SiLU(),  # 8
            nn.ConvTranspose2d(ch * 4, ch * 2, 4, 2, 1), nn.SiLU(),  # 16
            nn.ConvTranspose2d(ch * 2, ch, 4, 2, 1), nn.SiLU(),      # 32
            nn.ConvTranspose2d(ch, 1, 4, 2, 1),                      # 64, logits
        )

    def forward(self, z):
        return self.net(self.fc(z).view(-1, self.ch * 4, 4, 4))


def mlp(inp, out, hidden=256):
    return nn.Sequential(nn.Linear(inp, hidden), nn.SiLU(), nn.Linear(hidden, out))


class WorldModelV0(nn.Module):
    def __init__(self, latent_dim: int = 32, hidden_dim: int = 256, ch: int = 32):
        super().__init__()
        self.latent_dim, self.hidden_dim = latent_dim, hidden_dim
        self.encoder = Encoder(latent_dim, ch)
        self.decoder = Decoder(latent_dim, ch)
        self.gru = nn.GRUCell(latent_dim + NUM_ACTIONS, hidden_dim)
        self.prior = mlp(hidden_dim, latent_dim)
        self.reward_head = mlp(hidden_dim, 3)    # classes: -1, 0, +1
        self.continue_head = mlp(hidden_dim, 1)  # logit of "episode continues"

    def encode(self, obs):
        """obs: B,T,1,H,W -> B,T,D, each latent normalised to zero mean, unit variance."""
        B, T = obs.shape[:2]
        # Normalising keeps the latent's scale fixed: unbounded, the encoder inflates it and
        # the latent-matching loss explodes; squashed with tanh, it saturates and collapses.
        return F.layer_norm(self.encoder(obs.flatten(0, 1)), (self.latent_dim,)).view(B, T, -1)

    def decode(self, z):
        """z: B,T,D -> logits B,T,1,H,W"""
        B, T = z.shape[:2]
        return self.decoder(z.flatten(0, 1)).view(B, T, 1, 64, 64)

    def step(self, h, z, action):
        return self.gru(torch.cat([z, F.one_hot(action, NUM_ACTIONS).float()], -1), h)

    def rollout(self, z_real, actions, context: int):
        """Run the dynamics over a sequence.

        For t < context the GRU sees the encoder's latent (teacher forcing); afterwards
        it sees its own prediction (open loop / imagination).

        z_real: B,T,D (only the first `context` steps are used as inputs)
        actions: B,T
        Returns z_pred B,T-1,D (predictions for steps 1..T-1) and h B,T-1,H.
        """
        B, T, _ = z_real.shape
        h = z_real.new_zeros(B, self.hidden_dim)
        z_in = z_real[:, 0]
        z_preds, hs = [], []
        for t in range(T - 1):
            h = self.step(h, z_in, actions[:, t])
            z_hat = F.layer_norm(self.prior(h), (self.latent_dim,))
            z_preds.append(z_hat)
            hs.append(h)
            z_in = z_real[:, t + 1] if t + 1 < context else z_hat
        return torch.stack(z_preds, 1), torch.stack(hs, 1)


def reward_to_class(r):
    return (torch.sign(r) + 1).long()  # -1 -> 0, 0 -> 1, +1 -> 2


def class_to_reward(c):
    return c.float() - 1


def recon_loss(logits, target, foreground_weight: float = 20.0):
    """Bernoulli log-likelihood summed over pixels, with ball and paddle pixels upweighted.

    The ball covers ~20 of 4096 pixels. With an unweighted loss the autoencoder settles on
    "predict a black screen with a faint paddle" and never leaves that plateau (tested:
    no progress after 1200 steps). Upweighting foreground pixels breaks the plateau within
    ~500 steps. This is the "small objects get blurred away" pitfall from the primer.
    """
    weight = 1 + (foreground_weight - 1) * (target > 0.05).float()
    nll = F.binary_cross_entropy_with_logits(logits, target, weight=weight, reduction="none")
    return nll.sum((-3, -2, -1)).mean()


# Hits, misses and episode ends are each only ~2% of steps; upweight them so the heads
# do not just learn "nothing happens".
REWARD_CLASS_WEIGHTS = torch.tensor([10.0, 1.0, 10.0])
DONE_WEIGHT = 10.0


def compute_loss(model: WorldModelV0, batch: dict, context: int, latent_weight: float = 1.0):
    obs, act = batch["obs"], batch["action"]
    z = model.encode(obs)
    z_pred, h = model.rollout(z, act, context)

    # 1. Autoencoder: every real frame reconstructs from its own latent.
    loss_ae = recon_loss(model.decode(z), obs)
    # 2. Dynamics in pixel space: predicted latents must decode to the true next frames,
    #    both in the teacher-forced part and in the open-loop part.
    loss_pred = recon_loss(model.decode(z_pred), obs[:, 1:])
    # 3. Dynamics in latent space: match the encoder's latent (stop-gradient target).
    loss_latent = F.mse_loss(z_pred, z[:, 1:].detach())
    # 4. Heads: reward and termination for the transition that produced frame t+1.
    loss_reward = F.cross_entropy(
        model.reward_head(h).flatten(0, 1), reward_to_class(batch["reward"][:, :-1]).flatten(),
        weight=REWARD_CLASS_WEIGHTS.to(h.device),
    )
    done = batch["done"][:, :-1]
    loss_cont = F.binary_cross_entropy_with_logits(
        model.continue_head(h).squeeze(-1), 1 - done, weight=1 + (DONE_WEIGHT - 1) * done
    )
    total = loss_ae + loss_pred + latent_weight * loss_latent + loss_reward + loss_cont
    return total, {
        "loss": total.item(), "ae": loss_ae.item(), "pred": loss_pred.item(),
        "latent": loss_latent.item(), "reward": loss_reward.item(), "cont": loss_cont.item(),
    }
