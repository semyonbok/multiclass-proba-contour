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
# from sklearn.linear_model import LogisticRegression
# from sklearn.datasets import load_iris
# from sklearn.datasets import make_classification

plt.style.use("seaborn-ticks")

# TODO: docstrings, data generation, contour customisation, legend, streamlit


# %% the calss
class ProbaVis():
    """
    <>.

    ...

    Attributes
    ----------
    model :
        <>.
    train_data :
        <>.
    train_target :
        <>.

    Methods
    -------
    set_model(model)
        <>.
    set_data(train_data, train_target, features, grid_res=(100, 100))
        <>.
    plot()
        <>.
    replot()
        <>.

    """

    def __init__(
            self, model, train_data: pd.DataFrame, train_target: iter,
            features: iter, grid_res: tuple = (100, 100)
            ):
        self._define_utilities()
        self.set_model(model)
        self.set_data(train_data, train_target, features, grid_res)

    def _define_utilities(self):
        self._asserts = {
            "a1": "data & target must have the same length",
            "a2": "two features must be specified for visualization",
            "a3": "two integers must be used to specify grid resolution",
            "a4": "feature {} is not numeric"
            }
        self._colors = [
                "tab:blue", "tab:orange", "tab:green",
                "tab:red", "tab:purple", "tab:grey"
             ]

        self._cmap_colors = [
            "Blues", "Oranges", "Greens", "Reds", "Purples", "Greys"
            ]

        self._mstyles = ["o", "s", "P", "v", "D", "X"]

    def set_model(self, new_model):
        self.model = new_model

    def set_data(
            self, train_data: pd.DataFrame, train_target: iter,
            features: iter, grid_res: tuple = (100, 100)
            ):
        # input validation
        assert train_data.shape[0] == len(train_target), self._asserts["a"]
        assert len(features) == 2, self._asserts["a2"]
        assert len(grid_res) == 2 and all(
            [isinstance(res, int) for res in grid_res]
            ), self._asserts["a3"]

        try:
            train_data = train_data.iloc[:, features]
        except IndexError:
            train_data = train_data.loc[:, features]  # KeyError possible

        for feature in train_data.columns:
            assert pd.api.types.is_numeric_dtype(train_data[feature]) &\
                ~pd.api.types.is_bool_dtype(train_data[feature]),\
                self._asserts["a4"].format(feature)

        # define new entries for contour, ensure all data points will be seen
        coord_dict = {}
        for axis, feature in zip(["x", "y"], [0, 1]):
            coord_dict[axis] = np.linspace(
                train_data.iloc[:, feature].min() -
                train_data.iloc[:, feature].values.ptp()/100,
                train_data.iloc[:, feature].max() +
                train_data.iloc[:, feature].values.ptp()/100,
                grid_res[feature]
                )
        coord_dict["x"], coord_dict["y"] = np.meshgrid(
            coord_dict["x"], coord_dict["y"]
            )

        # set data attributes
        self._coord_dict = coord_dict
        self._mesh_entries = np.append(
            coord_dict["x"].reshape(coord_dict["x"].size, 1),
            coord_dict["y"].reshape(coord_dict["y"].size, 1),
            axis=1
            )
        self.train_data = train_data
        self.train_target = train_target

    def plot(self, fig_size=(10, 7), return_fig=False):
        # get predictions
        self.model.fit(self.train_data.values, self.train_target)
        pred_proba = self.model.predict_proba(self._mesh_entries)
        pred_class = self.model.predict(self._mesh_entries)
        train_score = self.model.score(
            self.train_data.values, self.train_target)
        full_data = self.train_data.assign(class_=self.train_target)

        # figure canvas and appearance
        fig, axes = plt.subplots(1, 1, figsize=fig_size, tight_layout=True)
        axes.set_title(f'{repr(self.model)}')
        axes.set_xlabel(self.train_data.iloc[:, 0].name, fontsize="large")
        axes.set_ylabel(self.train_data.iloc[:, 1].name, fontsize="large")
        axes.set_facecolor("k")  # for better decision boundary display

        # engage utilities
        cmap_cycle = cycle(self._cmap_colors)
        color_cycle = cycle(self._colors)
        marker_cycle = cycle(self._mstyles)

        # iteratively plot contours and data points for every class
        for index, class_ in enumerate(self.model.classes_):
            # main filled contour
            cs0 = axes.contourf(
                self._coord_dict["x"], self._coord_dict["y"], np.where(
                    (pred_class == class_), pred_proba[:, index], np.nan
                    ).reshape(
                        self._coord_dict["x"].shape[0],
                        self._coord_dict["y"].shape[1]
                        ),
                cmap=next(cmap_cycle), alpha=1,
                )

            # isolines
            cs1 = axes.contour(cs0, levels=cs0.levels[::2], colors="k")
            axes.clabel(cs1, cs1.levels, inline=True,)

            # data points
            axes.scatter(
                full_data.columns[0], full_data.columns[1],
                data=full_data.loc[full_data.class_ == class_],
                c=next(color_cycle), marker=next(marker_cycle), edgecolor="k",
                zorder=2, label=class_
                )

            axes.legend(
                loc="center left", bbox_to_anchor=(1.04, .5),
                title=f"Train score={train_score:.3f}\nClasses",
                fontsize="large", title_fontsize="large"
                )

        if return_fig:
            return fig

    # for widget
    def replot(self, **params):
        self.set_model(self.model.set_params(**params))
        self.plot()

# %% deprecated
# # %% data and model
# data, target = load_iris(return_X_y=True, as_frame=True)
# model = LogisticRegression()

# # after data upload, allow to select any two numerical features
# num_cols = data.columns[
#     [pd.api.types.is_numeric_dtype(data[x]) for x in data.columns]
#     ]
# data = data[num_cols].iloc[:, [0, 1]]

# # find a two-dimensional matrix with uniformly distributed feature values
# x_range = np.linspace(data.iloc[:, 0].min(), data.iloc[:, 0].max(), 100)
# y_range = np.linspace(data.iloc[:, 1].min(), data.iloc[:, 1].max(), 100)
# xx, yy = np.meshgrid(x_range, y_range)
# mesh_entries = np.append(
#     xx.reshape(xx.size, 1), yy.reshape(yy.size, 1), axis=1
#     )

# # %% plotting
# # TODO represent as a function; hyperparam to be changed with slider
# for param in np.logspace(-3, 3, num=7):  # range(1, 15, 2)
#     # get predictions
#     model.set_params({"C":param})
#     model.fit(data.values, target)
#     pred_proba = model.predict_proba(mesh_entries)
#     pred_class = model.predict(mesh_entries)
#     train_score = model.score(data.values, target)

#     # figure canvas and appearance
#     fig, ax = plt.subplots(1, 1, figsize=(10, 10))
#     ax.set_title(f'{repr(model)} | train score = {train_score:.3f}')
#     ax.set_xlabel(data.iloc[:, 0].name)
#     ax.set_ylabel(data.iloc[:, 1].name)

#     for i, (c, cmap) in enumerate(
#             # TODO add colormap cycler
#             zip(model.classes_, ["Reds", "Greens", "Blues"])
#             ):
#         # main filled contour
#         cs0 = ax.contourf(
#             xx, yy, np.where(
#                 (pred_class == c), pred_proba[:, i], np.nan
#                 ).reshape(xx.shape),
#             cmap=cmap, alpha=.5,
#             )
#         cs1 = ax.contour(cs0, levels=cs0.levels[::2], colors="k")

#         ax.clabel(cs1, cs1.levels, inline=True,)

#         # decision boundary
#         cs2 = ax.contour(
#             xx, yy, pred_class.reshape(xx.shape),
#             colors="k", linewidths=.5
#             )

#     # TODO use plt.plot to enable labelling, add color cycler
#     ax.scatter(
#         data.iloc[:, 0], data.iloc[:, 1],
#         c=target.map({0: "tab:red", 1: "tab:green", 2: "tab:blue"}),
#         edgecolor="k"
#         )
