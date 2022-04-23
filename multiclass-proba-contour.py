# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 19:42:13 2022

@author: Semyon
"""
# %% set-up
# pylint: disable=fixme

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
# from sklearn.datasets import

plt.style.use("seaborn-ticks")
plt.rcParams['font.size'] = 10


# %% data and model
data, target = load_iris(return_X_y=True, as_frame=True)
model = KNeighborsClassifier()

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
# TODO represent as a function
for param in range(1,15,2):  # np.logspace(-3, 3, num=7):  # to be changed with slider
    # get predictions
    model.set_params(n_neighbors=param)
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
