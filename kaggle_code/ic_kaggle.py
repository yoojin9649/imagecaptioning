import wandb

import tensorflow as tf
from tensorflow.keras.layers import *

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import IPython

# Data
data_dir='./flickr30k_images/flickr30k_images'
image_dir=f'{data_dir}/flickr30k_images'
csv_file=f'{data_dir}/results.csv'

df=pd.read_csv(csv_file, delimiter='|')

print(f'[INFO] The shape of dataframe: {df.shape}')
print(f'[INFO] The columns in the dataframe: {df.columns}')
print(f'[INFO] Unique image names: {len(pd.unique(df["image_name"]))}')

df.head(5)

df.colums=['image_name']