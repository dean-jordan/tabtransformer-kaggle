import torch
import torch.nn as nn
import pytorch_tabular
import pandas as pd
from sklearn.preprocessing import StandardScaler

from pytorch_tabular.models.tab_transformer import TabTransformerConfig
from pytorch_tabular import TabularModel
from pytorch_tabular.config import DataConfig

train_data_path = '../data/train.csv'

# Opening the dataset
train_data = pd.read_csv(train_data_path)

# Applying StandardScaler
normalized_scalers = StandardScaler()
preprocessed_data = normalized_scalers.fit_transform(train_data)

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