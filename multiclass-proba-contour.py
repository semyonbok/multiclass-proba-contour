# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 19:42:13 2022

@author: Semyon
"""
# %% set-up
# pylint: disable=fixme

from itertools import cycle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
# from sklearn.datasets import make_classification

plt.style.use("seaborn-ticks")
plt.se


# %% the calss
class ProbaVis():
    def __init__(
            self, train_data: pd.DataFrame, train_target: iter,
            features: iter, grid_res: tuple = (100, 100)
            ):
        self._define_utilities()
        self.set_data(train_data, train_target, features, grid_res)

    def _define_utilities(self):
        self._asserts = {
            "a1": "data & target must have the same length",
            "a2": "two features must be specified",
            "a3": "feature {} is not numeric"
            }
        self._cmaps = cycle(
            ["Blues", "Oranges", "Greens", "Reds", "Purples",
             "YlOrBr"]
            )
        
    def set_data(
            self, train_data: pd.DataFrame, train_target: iter,
            features: iter, grid_res: tuple = (100, 100)
            ):
        # input validation
        assert train_data.shape[0] == len(train_target), self._asserts["a1"]
        assert len(features) == 2, self._asserts["a2"]

        try:
            train_data = train_data.iloc[:, features]
        except IndexError:
            train_data = train_data.loc[:, features]  # KeyError possible

        for feature in train_data.columns:
            assert pd.api.types.is_numeric_dtype(train_data[feature]) &\
                ~pd.api.types.is_bool_dtype(train_data[feature]),\
                self._asserts["a3"].format(feature)

        # define all new entries for contour
        range_dict = {}
        for axis, f in zip(["x", "y"], [0, 1]):
            range_dict[axis] = np.linspace(
                train_data.iloc[:, f].min() - train_data.iloc[:, f].std()/100,
                train_data.iloc[:, f].max() + train_data.iloc[:, f].std()/100,
                grid_res[f]
                )
        xx, yy = np.meshgrid(range_dict["x"], range_dict["y"])
        self._mesh_entries = np.append(
            xx.reshape(xx.size, 1), yy.reshape(yy.size, 1), axis=1
            )
        self.train_data = train_data
        self.train_target = train_target


# %% data and model
data, target = load_iris(return_X_y=True, as_frame=True)
model = LogisticRegression()

# after data upload, allow to select any two numerical features
num_cols = data.columns[
    [pd.api.types.is_numeric_dtype(data[x]) for x in data.columns]
    ]
data = data[num_cols].iloc[:, [0, 1]]

# find a two-dimensional matrix with uniformly distributed feature values
x_range = np.linspace(data.iloc[:, 0].min(), data.iloc[:, 0].max(), 100)
y_range = np.linspace(data.iloc[:, 1].min(), data.iloc[:, 1].max(), 100)
xx, yy = np.meshgrid(x_range, y_range)
mesh_entries = np.append(
    xx.reshape(xx.size, 1), yy.reshape(yy.size, 1), axis=1
    )

# %% plotting
# TODO represent as a function; hyperparam to be changed with slider
for param in np.logspace(-3, 3, num=7):  # range(1, 15, 2)
    # get predictions
    model.set_params(C=param)
    model.fit(data.values, target)
    pred_proba = model.predict_proba(mesh_entries)
    pred_class = model.predict(mesh_entries)
    train_score = model.score(data.values, target)

    # figure canvas and appearance
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.set_title(f'{repr(model)} | train score = {train_score:.3f}')
    ax.set_xlabel(data.iloc[:, 0].name)
    ax.set_ylabel(data.iloc[:, 1].name)

    for i, (c, cmap) in enumerate(
            # TODO add colormap cycler
            zip(model.classes_, ["Reds", "Greens", "Blues"])
            ):
        # main filled contour
        cs0 = ax.contourf(
            xx, yy, np.where(
                (pred_class == c), pred_proba[:, i], np.nan
                ).reshape(xx.shape),
            cmap=cmap, alpha=.5,
            )
        cs1 = ax.contour(cs0, levels=cs0.levels[::2], colors="k")

        ax.clabel(cs1, cs1.levels, inline=True,)

        # decision boundary
        cs2 = ax.contour(
            xx, yy, pred_class.reshape(xx.shape),
            colors="k", linewidths=.5
            )

    # TODO use plt.plot to enable labelling, add color cycler
    ax.scatter(
        data.iloc[:, 0], data.iloc[:, 1],
        c=target.map({0: "tab:red", 1: "tab:green", 2: "tab:blue"}),
        edgecolor="k"
        )
