import param
import panel as pn

from sklearn.datasets import load_iris, load_wine

import multiclass_proba_contour as mpc

pn.extension(template="material")
# panel serve ProbaVis_panel_app.py --show --autoreload

# Widget options
all_datasets = ["Iris", "Wine"]
all_models = ["K Nearest Neighbors", "Random Forest"]
iris_features = [
    'sepal length (cm)', 'sepal width (cm)',
    'petal length (cm)', 'petal width (cm)'
    ]

wine_features = [
    'alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium',
    'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins',
    'color_intensity', 'nonflavanoid_phenols', 'proanthocyanins',
    'color_intensity', 'hue', 'od280/od315_of_diluted_wines', 'proline'
]

# Widgets with deault values
data_widget = pn.widgets.Select(
    name="Select one of the data sets", options=all_datasets, value="Iris"
    )
f1_widget = pn.widgets.Select(
    name="Pick the first feature", options=iris_features, value=iris_features[0]
    )
f2_widget = pn.widgets.Select(
    name="Pick the second feature", options=iris_features[1:], value=iris_features[1]
    )
model_widget = pn.widgets.Select(
    name="Select one of the models", options=all_models
    )

# Callback functions that updates the downstream widgets
def update_features(event):
    if event.new == "Iris":
        f1_widget.options = iris_features
        f2_widget.options = iris_features[1:]
    elif event.new == "Wine":
        f1_widget.options = wine_features
        f2_widget.options = wine_features[1:]
 
def update_f2(event):
    f2_options = f1_widget.options.copy()
    f2_options.remove(event.new)
    f2_widget.options = f2_options

knn_hyper_widgets = {
    "n_neighbors" : pn.widgets.IntSlider(
        name="n_neighbors", value=5, start=1, end=100),
    "algorithm" : pn.widgets.Select(
        name="algorithm", options=['auto', 'ball_tree', 'kd_tree', 'brute']),
        }

class HyperWidgetHolder(param.Parameterized):
    hyper_widgets = param.Parameter(default=knn_hyper_widgets)

holder = HyperWidgetHolder()

def create_hyper_widgets(event):
    if event.new == "K Nearest Neighbors":
        holder.hyper_widgets = knn_hyper_widgets
    elif event.new == "Random Forest":
        holder.hyper_widgets = {
            "criterion" : pn.widgets.Select(
                name="criterion", options=['gini', 'entropy']),
            "n_estimators" : pn.widgets.IntSlider(
                name="n_estimators", value=100, start=1, end=500),
        }

# Attach the callbacks to the 'value' parameter of appropriate widgets
data_widget.param.watch(update_features, 'value')
f1_widget.param.watch(update_f2, "value")
model_widget.param.watch(create_hyper_widgets, "value")

input_column=pn.Column()
input_column.extend([
    pn.pane.Markdown("## Data & Features"),
    data_widget, f1_widget, f2_widget,
    pn.pane.Markdown("## Model & Hyperparameters"),
    ])

def master_func(set_name, model, f1, f2, **model_params):
    if set_name == "Wine":
        data_set = load_wine(as_frame=True)
    elif set_name == "Iris":
        data_set = load_iris(as_frame=True)
    target_names_map = {
        k: v for k, v in zip(
            range(data_set["target"].nunique()), data_set["target_names"]
        )
    }
    data_set["target"] = data_set["target"].map(target_names_map)
    data = data_set["data"]
    target = data_set["target"]

    if model == "K Nearest Neighbors":
        from sklearn.neighbors import KNeighborsClassifier
        model = KNeighborsClassifier()
        
    if model == "Random Forest":
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier()

    model.set_params(**model_params)
    pv = mpc.ProbaVis(model, data, target, [f1, f2])
    fig = pv.plot(return_fig=True, fig_size=(9, 6))
    predict = pv.model.predict(data[[f1, f2]].values)

    accuracy_per_class = {}
    dials = []
    for name, dial_color in zip(
            data_set["target_names"],
            [
                [(0.2, '#d0e1f2'), (0.4, '#94c4df'), (0.6, '#4a98c9'), (0.8, '#1764ab'), (1.0, '#08306b')],
                [(0.2, '#fdd9b4'), (0.4, '#fda762'), (0.6, '#f3701b'), (0.8, '#c54102'), (1.0, '#7f2704')],
                [(0.2, '#d3eecd'), (0.4, '#98d594'), (0.6, '#4bb062'), (0.8, '#157f3b'), (1.0, '#00441b')]
            ]
        ):
        name_mask = target.values == name
        accuracy_per_class[name] = (predict[name_mask] == target.values[name_mask]).sum()
        accuracy_per_class[name] *= 100/len(target.values[name_mask])
        dials.append(pn.indicators.Dial(
            name=name, value=accuracy_per_class[name].round(2), bounds=(0,100),
            colors=dial_color,
            ))

    # conditional operator to prevent dial aliasing
    if set_name == "Iris":
        return pn.Column(
            pn.pane.Matplotlib(fig, format='svg',),
            pn.Row(*dials, align="center"),
        )
    return pn.Column(
        pn.pane.Matplotlib(fig, format='svg',),
        pn.Row(),
        pn.Row(*dials, align="center"),
        )

# @pn.depends(holder.param.hyper_widgets)
# def display_widget(widget):
#     return widget
# pn.Column(pn.Column(display_widget)).servable()

master_bind = pn.bind(
    master_func, set_name=data_widget, model=model_widget,
    f1=f1_widget, f2=f2_widget, #**holder.hyper_widgets
    )

hyper_bind = pn.bind(lambda x: pn.Column(*holder.hyper_widgets.values()), model_widget)

# app layout
pn.Row(master_bind).servable(title="Welcome to Multiclass Probability Visualizer!")
hyper_column = pn.Column(model_widget, hyper_bind)
input_column.servable(target="sidebar")
hyper_column.servable(target="sidebar")
