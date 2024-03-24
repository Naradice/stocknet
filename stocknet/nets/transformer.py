import math

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import TransformerDecoder, TransformerDecoderLayer, TransformerEncoder, TransformerEncoderLayer


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.05, batch_first=True, device=None, **kwargs):
        super().__init__()
        self.args = {"d_model": d_model, "max_len": max_len, "dropout": dropout}
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model, device=device)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(-2)
        if batch_first:
            pe = pe.transpose(0, 1)
            self.forward = self.__fforward
        else:
            self.forward = self.__mforward

        self.register_buffer("pe", pe)

    def __mforward(self, src, tgt):
        src_pos = src.size(0)
        tgt_pos = src_pos + tgt.size(0) - 1
        return self.dropout(src + self.pe[:src_pos, :]), self.dropout(tgt + self.pe[src_pos - 1 : tgt_pos, :])

    def __fforward(self, src, tgt):
        src_pos = src.size(1)
        tgt_pos = src_pos + tgt.size(1) - 1
        return self.dropout(src + self.pe[:, :src_pos, :]), self.dropout(tgt + self.pe[:, src_pos - 1 : tgt_pos, :])


class EmbeddingPositionalEncoding(nn.Module):
    def __init__(self, num_embedding, d_model, dropout=0.1, device=None, **kwargs):
        super().__init__()
        self.args = {"num_embedding": num_embedding, "d_model": d_model}
        self.pe = nn.Embedding(num_embedding, d_model, device=device)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, time_ids):
        position = self.pe(time_ids)
        position = self.dropout(position)
        return position


class Seq2SeqTransformer(nn.Module):
    def __init__(
        self,
        num_encoder_layers: int,
        num_decoder_layers: int,
        d_model: int,
        positional_encoding,
        output_layer=None,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        nhead: int = 8,
        batch_first=True,
        device=None,
        **kwargs,
    ):
        """Simple Transformer model

        Args:
            num_encoder_layers (int): number of encoder layer
            num_decoder_layers (int): number of decoder layer
            d_model (int): d_model size. if outputlayer is None, this become output size.
            positional_encoding (nn.Module): positional encoding
            output_layer (nn.Module, optional): convert layer from d_model to expected size. Defaults to None.
            dim_feedforward (int, optional): number of FF layer. Defaults to 512.
            dropout (float, optional): dropout ratio for all layers. Defaults to 0.1.
            nhead (int, optional): nhead size of attention. Defaults to 8.
            batch_first (bool, optional): if True, expect (batch_size, obs_length, feature_size). Defaults to True.
        """

        super(Seq2SeqTransformer, self).__init__()
        self.args = {
            "num_encoder_layers": num_encoder_layers,
            "num_decoder_layers": num_decoder_layers,
            "d_model": d_model,
            "dim_feedforward": dim_feedforward,
            "dropout": dropout,
            "nhead": nhead,
            "positional_encoding": {"key": positional_encoding._get_name(), **positional_encoding.args},
        }

        self.positional_encoding = positional_encoding
        self.output_layer = output_layer
        if output_layer is not None:
            args = {}
            if hasattr(output_layer, "args"):
                args = output_layer.args
            self.args["output_layer"] = {"key": output_layer._get_name(), **args}
        if isinstance(positional_encoding, PositionalEncoding):
            self.forward = self.__forward
        elif isinstance(positional_encoding, EmbeddingPositionalEncoding):
            self.forward = self.__time_forward
        else:
            raise ValueError(f"unsupported positional encoding: {type(positional_encoding)}")
        self.dropout = nn.Dropout(dropout)

        encoder_layer = TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=batch_first, device=device
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        decoder_layer = TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=batch_first, device=device
        )
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.transformer_decoder.forward()

    @classmethod
    def load(
        self,
        num_encoder_layers: int,
        num_decoder_layers: int,
        d_model: int,
        positional_encoding: dict,
        output_layer=None,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        nhead: int = 8,
        batch_first=True,
        device=None,
        **kwargs,
    ):
        pe = None
        positional_encoding = positional_encoding.copy()
        positional_encoding_key = positional_encoding.pop("key")
        if positional_encoding_key == "PositionalEncoding":
            if "dropout" not in positional_encoding:
                positional_encoding["dropout"] = dropout
            pe = PositionalEncoding(d_model=d_model, **positional_encoding, batch_first=batch_first, device=device)
        elif positional_encoding_key == "EmbeddingPositionalEncoding":
            if "dropout" not in positional_encoding:
                positional_encoding["dropout"] = dropout
            emmb_num = None
            if "num_embedding" in kwargs:
                emmb_num = kwargs["num_embedding"]
            if "num_embedding" in positional_encoding:
                emmb_num = positional_encoding.pop("num_embedding")
            if emmb_num is None:
                raise ValueError("num_embedding is not found in kwargs to create PositionalEncoding")
            else:
                positional_encoding["num_embedding"] = emmb_num
                pe = EmbeddingPositionalEncoding(d_model=d_model, device=device, **positional_encoding)
        else:
            raise ValueError(f"valid positional encoding is not specified: {positional_encoding_key}")
        model = Seq2SeqTransformer(
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            d_model=d_model,
            positional_encoding=pe,
            output_layer=output_layer,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            nhead=nhead,
            batch_first=batch_first,
            device=device,
        )
        return model

    def __forward(
        self,
        src: Tensor,
        tgt: Tensor,
        mask_tgt: Tensor,
        mask_src: Tensor = None,
        padding_mask_src: Tensor = None,
        padding_mask_tgt: Tensor = None,
        memory_key_padding_mask: Tensor = None,
    ):
        src, tgt = self.positional_encoding(src, tgt)
        memory = self.transformer_encoder(src, mask_src, padding_mask_src)
        outs = self.transformer_decoder(tgt, memory, mask_tgt, None, padding_mask_tgt, memory_key_padding_mask)
        if self.output_layer is not None:
            outs = self.output_layer(outs)
        return outs

    def __time_forward(
        self,
        src: Tensor,
        src_time: Tensor,
        tgt: Tensor,
        tgt_time: Tensor,
        mask_tgt: Tensor,
        mask_src: Tensor = None,
        padding_mask_src: Tensor = None,
        padding_mask_tgt: Tensor = None,
        memory_key_padding_mask: Tensor = None,
    ):
        src_time = self.positional_encoding(src_time)
        src = self.dropout(torch.add(src, src_time))
        tgt_time = self.positional_encoding(tgt_time)
        tgt = self.dropout(torch.add(tgt, tgt_time))
        memory = self.transformer_encoder(src, mask_src, padding_mask_src)
        outs = self.transformer_decoder(tgt, memory, mask_tgt, None, padding_mask_tgt, memory_key_padding_mask)
        if self.output_layer is not None:
            outs = self.output_layer(outs)
        return outs
