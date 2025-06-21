import torch
import torch.nn as nn
import pytorch_tabular

from pytorch_tabular.models.tab_transformer import TabTransformerConfig
from pytorch_tabular import TabularModel
from pytorch_tabular.config import DataConfig

data_configuration = DataConfig(
    target = [
        'Fertilizer Name'
    ],

    continuous_cols = [
        'Temperature',
        'Humidity',
        'Moisture',
        'Nitrogen',
        'Potassium',
        'Phosphorous'
    ],

    categorical_cols = [
        'Soil Type',
        'Crop Type'
    ]
)