import torch.nn as nn

from .transformer import EmbeddingPositionalEncoding, PositionalEncoding, Seq2SeqTransformer


class MeanVarianceLSTM(nn.Module):
    """LSTM that jointly predicts expected return (mean) and log-variance for each feature."""

    def __init__(self, input_dim, hidden_dim, output_dim, device, batch_first=True, **kwargs):
        super().__init__()
        self.args = {"input_dim": input_dim, "hidden_dim": hidden_dim, "output_dim": output_dim}
        self.rnn = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=batch_first, device=device)
        self.mean_layer = nn.Linear(hidden_dim, output_dim, device=device)
        self.log_var_layer = nn.Linear(hidden_dim, output_dim, device=device)
        self.device = device

    def forward(self, inputs, hidden0=None):
        output, _ = self.rnn(inputs.to(self.device), hidden0)
        hidden = output[:, -1, :]
        mean = self.mean_layer(hidden)
        log_var = self.log_var_layer(hidden)
        return mean, log_var


class MeanVarianceTransformer(nn.Module):
    """Seq2Seq Transformer that jointly predicts expected return (mean) and log-variance per timestep."""

    def __init__(self, transformer: Seq2SeqTransformer, mean_head: nn.Linear, log_var_head: nn.Linear, output_dim: int, **kwargs):
        super().__init__()
        self.args = {**transformer.args, "output_dim": output_dim}
        self.transformer = transformer
        self.mean_head = mean_head
        self.log_var_head = log_var_head

    @classmethod
    def load(
        cls,
        num_encoder_layers: int,
        num_decoder_layers: int,
        d_model: int,
        positional_encoding: dict,
        output_dim: int,
        input_layer=None,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        nhead: int = 8,
        batch_first: bool = True,
        device=None,
        **kwargs,
    ):
        inner = Seq2SeqTransformer.load(
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            d_model=d_model,
            positional_encoding=positional_encoding.copy(),
            input_layer=input_layer,
            output_layer=None,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            nhead=nhead,
            batch_first=batch_first,
            device=device,
            **kwargs,
        )
        mean_head = nn.Linear(d_model, output_dim, device=device)
        log_var_head = nn.Linear(d_model, output_dim, device=device)
        return cls(inner, mean_head, log_var_head, output_dim)

    def forward(self, *args, **kwargs):
        hidden = self.transformer(*args, **kwargs)
        mean = self.mean_head(hidden)
        log_var = self.log_var_head(hidden)
        return mean, log_var
