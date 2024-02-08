import torch
from torch import nn
from dataclasses import dataclass
import os
import pytest
from pprint import pprint
from conditional_rate_matching.data.music_dataloaders import LankhPianoRollDataloader
from conditional_rate_matching.data.music_dataloaders_config import LakhPianoRollConfig

from conditional_rate_matching.models.generative_models.crm import CRM
from conditional_rate_matching.models.temporal_networks.temporal_networks_config import TemporalDeepMLPConfig
from conditional_rate_matching.data.states_dataloaders_config import StatesDataloaderConfig
from conditional_rate_matching.configs.configs_classes.config_crm import CRMConfig, CRMTrainerConfig, \
    BasicPipelineConfig
from conditional_rate_matching.models.temporal_networks.temporal_networks_config import DiffusersUnet2DConfig
from conditional_rate_matching.data.music_dataloaders_config import LakhPianoRollConfig
from conditional_rate_matching.models.temporal_networks.temporal_embedding_utils import transformer_timestep_embedding

@dataclass
class TemporalTransformerConfig:
    nhead: int              # Number of attention heads
    num_encoder_layers: int # Number of encoder layers
    num_decoder_layers: int # Number of decoder layers
    dim_feedforward: int    # Dimension of feedforward network
    dropout: float          # Dropout rate
    activation: str         # Activation function of the encoder/decoder intermediate layer
    pad_token_id: int       # Padding token id for batch processing
    time_embed_dim:int = 19


def CustomCombine(src_emb, continuous_emb):
    # Check shapes
    assert src_emb.shape[0] == continuous_emb.shape[0]  # batch size must be the same
    assert src_emb.shape[2] == continuous_emb.shape[2]  # embedding dimension must be the same

    # Concatenate along the sequence length dimension
    combined_emb = torch.cat([continuous_emb, src_emb], dim=1)
    return combined_emb

# Example configuration
class MyTransformerWithConditioning(nn.Module):
    def __init__(self, config: CRMConfig):
        super(MyTransformerWithConditioning, self).__init__()
        self.config = config

        self.vocab_size = config.data1.vocab_size
        self.d_model = config.data1.dimensions
        self.continuous_dim = config.temporal_network.time_embed_dim

        self.embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.continuous_embedding = nn.Linear(self.continuous_dim, self.d_model)
        self.positional_encoding = nn.Parameter(torch.zeros(1, self.d_model))

        # Consider a custom temporal encoding if needed
        # self.temporal_encoding = CustomTemporalEncoding(self.d_model, ...)

        self.transformer = nn.Transformer(
            d_model=self.d_model,
            nhead=config.temporal_network.nhead,
            num_encoder_layers=config.temporal_network.num_encoder_layers,
            num_decoder_layers=config.temporal_network.num_decoder_layers,
            dim_feedforward=config.temporal_network.dim_feedforward,
            dropout=config.temporal_network.dropout,
            activation=config.temporal_network.activation
        )
        self.out = nn.Linear(self.d_model, self.vocab_size)

    def forward(self, src, continuous_vector):
        src_emb = self.embedding(src) + self.positional_encoding
        continuous_emb = self.continuous_embedding(continuous_vector).unsqueeze(1)

        # Adjust the continuous embedding shape or the way of combining
        combined_emb = CustomCombine(src_emb, continuous_emb)

        output = self.transformer(combined_emb, combined_emb)
        return self.out(output)


# Example usage
continuous_dim = 100  # dimension of your continuous vector


if __name__=="__main__":

    epochs = 10
    batch_size = 32
    config = CRMConfig()
    config.data0 = LakhPianoRollConfig(batch_size=batch_size,
                                       conditional_model=True,
                                       bridge_conditional=True)
    config.data1 = config.data0

    config.trainer = CRMTrainerConfig(
        number_of_epochs=epochs,
        learning_rate=1e-4,
        metrics=[]
    )

    config.pipeline = BasicPipelineConfig(number_of_steps=5)
    config.temporal_network = TemporalDeepMLPConfig()
    crm = CRM(config=config, device=torch.device("cpu"))

    # CREATE THE TRANSFORMER
    temporal_config = TemporalTransformerConfig(
        nhead=2,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=2048,
        dropout=0.1,
        activation='relu',
        pad_token_id=0,  # assuming 0 is the pad token id
        time_embed_dim=9
    )


    config.temporal_network = temporal_config
    data_0, data_1 = next(crm.parent_dataloader.train().__iter__())
    times = torch.rand(data_0[0].size(0))
    model = MyTransformerWithConditioning(config)
    time_embeddings = transformer_timestep_embedding(times, embedding_dim=config.temporal_network.time_embed_dim)

    out = model(data_0[0].long(),time_embeddings)
    print(out.shape)
