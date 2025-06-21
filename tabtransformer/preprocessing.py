import torch
import torch.nn as nn
import pytorch_tabular
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from pytorch_tabular.models.tab_transformer import TabTransformerConfig
from pytorch_tabular import TabularModel
from pytorch_tabular.config import DataConfig

train_data_path = '../data/train.csv'

# Opening the dataset
total_data = pd.read_csv(train_data_path)

# Applying StandardScaler
normalized_scalers = StandardScaler()
total_data = normalized_scalers.fit_transform(total_data)

# Applying train-test split
train_data, test_data = train_test_split(
    total_data,
    test_size=0.15,
    random_state=42
)

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