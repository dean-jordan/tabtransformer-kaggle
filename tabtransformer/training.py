import torch
import torch.nn as nn
import pytorch_tabular
import pandas as pd

from pytorch_tabular.models.tab_transformer import TabTransformerConfig
from pytorch_tabular.config import OptimizerConfig
from pytorch_tabular.config import TrainerConfig

from model import tab_transformer_model
from preprocessing import preprocessed_data

training_configuration = TrainerConfig(
    batch_size = 64,
    max_epochs = 200
)

# Defaults to Adam Optimizer
optimizer_configuration = OptimizerConfig()

tab_transformer_model.fit(train=preprocessed_data)