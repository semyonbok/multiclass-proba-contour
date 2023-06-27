import panel as pn
from sklearn.datasets import load_iris, load_wine

import multiclass_proba_contour as mpc

pn.extension(template="material")
# panel serve ProbaVis_panel_app_master.py --show --autoreload

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
update_button = pn.widgets.Button(name='Update plot', button_type='primary')
model_widget = pn.widgets.Select(name="Select one of the models", options=all_models)

# Define a callback functions that updates the options of f1_widget and f2_widget
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

# Attach the callbacks to the 'value' parameter
data_widget.param.watch(update_features, 'value')
f1_widget.param.watch(update_f2, "value")

input_column=pn.Column()
input_column.extend([data_widget, f1_widget, f2_widget, model_widget, update_button])

def master_func(set_name, model, f1, f2):
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

    # feature_pick = lambda f, default=0: f if f is not None else default
    pv = mpc.ProbaVis(model, data, target, [f1, f2])
    
    return pv.plot(return_fig=True, fig_size=(9, 6))

master_bind = pn.bind(
    master_func, set_name=data_widget, model=model_widget,
    f1=f1_widget, f2=f2_widget
    )

# Make the master_bind function depend on the click event of the update button
@pn.depends(update_button.param.clicks)
def update_plot(clicks):
    return master_bind()

app = pn.Row(
    input_column,
    pn.pane.Matplotlib(update_plot, format='svg', sizing_mode='scale_both'),
     ).servable(title="Welcome to Multiclass Probability Visualizer!")

# app = pn.Row(
#     input_column,
#     pn.pane.Matplotlib(master_bind, format='svg', sizing_mode='scale_both'),
#     # pn.pane.Matplotlib("https://scikit-learn.org/stable/_images/sphx_glr_plot_classifier_comparison_001.png")
#     ).servable(title="Welcome to Multiclass Probability Visualizer!")
