import torch
import torch.nn as nn
import pytorch_tabular

from preprocessing import data_configuration
from training import optimizer_configuration
from training import training_configuration
from training import experiment_configuration

from pytorch_tabular.models.tab_transformer import TabTransformerConfig
from pytorch_tabular import TabularModel
from sklearn.metrics import accuracy_score, f1_score, precision_score
from pytorch_tabular.models.common.heads import LinearHeadConfig

# Adjust to fit data
tab_transformer_configuration = TabTransformerConfig(
    task="classification",
    num_features=8,
    num_classes=9999999, # Cannot determine, will look
    embedding_dim=1024,
    num_heads=8,
    num_attn_blocks=8,
    num_transformer_layers=8,
    mlp_hidden_dims=[512, 256, 128, 64],
    loss=nn.PairwiseDistance(),
    metrics=['accuracy', 'f1_score', 'precision_score'],
    ff_dropout=0.5,
    attn_dropout=0.5,
    add_norm_dropout=0.5,
    embedding_dropout=0.5,
    input_embed_dim=1024,
    transformer_head_dim=1024,
    transformer_activation='ReLU'
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
    verbose = True,
    suppress_lightning_logger=True,
    experiment_config = experiment_configuration
)

# Mainly to set dropout
head_configuration = LinearHeadConfig(
    layers = '',
    dropout = 0.5,
    initialization = 'kaiming'
).__dict__