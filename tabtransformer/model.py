import torch
import torch.nn as nn
import pytorch_tabular

from preprocessing import data_configuration
from training import optimizer_configuration
from training import training_configuration

from pytorch_tabular.models.tab_transformer import TabTransformerConfig
from pytorch_tabular import TabularModel

# Adjust to fit data
tab_transformer_configuration = TabTransformerConfig(
    task="classification",
    num_features=10,
    num_classes=2,
    embedding_dim=8,
    num_heads=4,
    num_transformer_layers=2,
    mlp_hidden_dims=[64, 32],
)

class TabTransformerModel(nn.Module):
    def __init__(self, config):
        super(TabTransformerModel, self).__init__()
        self.model = TabularModel(config)

    def forward(self, x):
        return self.model(x)
    
tab_transformer_model = TabularModel(
    data_config = data_configuration,
    model_config = tab_transformer_configuration,
    optimizer_config = optimizer_configuration,
    trainer_config = training_configuration,
    verbose = True
)