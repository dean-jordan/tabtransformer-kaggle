import torch
import torch.nn as nn
import pytorch_tabular
import pandas as pd

from pytorch_tabular.models.tab_transformer import TabTransformerConfig
from pytorch_tabular.config import OptimizerConfig
from pytorch_tabular.config import TrainerConfig

from model import tab_transformer_model
from preprocessing import train_data, test_data

# Early stopping and checkpoint saving is enabled by default
training_configuration = TrainerConfig(
    batch_size = 64,
    max_epochs = 200,
    accelerator = 'gpu',
    min_epochs = 10,
    checkpoints_path = '../models/checkpoints',
    auto_lr_find = True
)

# Defaults to Adam Optimizer
optimizer_configuration = OptimizerConfig()

tab_transformer_model.fit(train=train_data, test=test_data)