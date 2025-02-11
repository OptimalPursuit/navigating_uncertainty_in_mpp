from torch import nn
from models.decoder import AttentionDecoderWithCache

class Autoencoder(nn.Module):
    """Autoencoder model that takes in an observation and returns the decoded output."""
    def __init__(self, encoder, decoder, env):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.env = env

    def forward(self, obs):
        hidden, init_h = self.encoder(obs)
        if isinstance(self.decoder, AttentionDecoderWithCache):
            _, _, hidden = self.decoder.pre_decoder_hook(obs, self.env, hidden)
        dec_out = self.decoder(obs, hidden)
        return dec_out
