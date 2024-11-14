import polars as pl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Dataset:
    def __init__(self, path):
        self.path = path
        self.data = self._load_data()

    def _load_data(self):
        return pl.read_csv(self.path).to_pandas()

    def get_data(self):
        return self.data

    def get_features(self):
        return self.data.drop(columns=['target'])

    def get_target(self):
        return self.data['target']

    def get_feature_names(self):
        return self.get_features().columns

    def get_target_name(self):
        return 'target'

    def get_feature_types(self):
        return self.get_features().dtypes

    def get_target_type(self):
        return self.get_target().dtype

    def get_feature_shapes(self):
        return self.get_features().shape

    def get_target_shape(self):
        return self.get_target().shape

    def get_feature_info(self):
        return self.get_features().info()

    def get_target_info(self):
        return self.get_target().info()

    def get_feature_describe(self):
        return self.get_features().describe()

    def get_target_describe(self):
        return self.get_target().describe()

    def get_feature_head(self, n=5):
        return self.get_features().head(n)

    def get_target_head(self, n=5):
        return self.get_target().head(n)

    def get_feature_tail(self, n=5):
        return self.get_features().tail(n)

    def get_target_tail(self, n=5):
        return self.get_target().tail(n)

    def plot_features(self):
        self.get_features().plot(kind='hist', subplots=True, layout=(3, 3), sharex=False, sharey=False)
        plt.show()

    def plot_target(self):
        self.get_target().plot(kind='hist')
        plt.show()