import re
import streamlit as st
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris, load_wine
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from src.multiclass_proba_contour import ProbaVis


# data processing functions
@st.cache_data
def process_toy(set_name):
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
    return data_set["data"], data_set["target"]

def fetch_model(model_pick):
    """Needed to avoid terminal error caused by model seltction checkbox"""
    if model_pick == "K Nearest Neighbors":
        return KNeighborsClassifier()
    if model_pick == "Random Forest":
        return RandomForestClassifier()
    if model_pick == "Gradient Boosting":
        return GradientBoostingClassifier()

def parse_param_desc(model):
    params = model.get_params().keys()
    params = "|".join([p + " : " for p in params])

    params_desc = re.split(params, model.__doc__)[1:]
    params_desc[-1] = params_desc[-1].split("Attributes\n")[0]
    params_desc = {k[:-3]: "\n".join(v.split("\n\n")) for k, v in zip(
        re.findall(params, model.__doc__), params_desc
    )}
    return params_desc


def none_or_widget(name, *wargs, widget=st.slider, **wkwargs):
    """"""
    name = " ".join(name.split("_"))
    if st.checkbox(
        "Set " + name,  # key=name,
        help="Default value is `None`. Select the checkbox to set another value."
    ):
        return widget(name.capitalize(), *wargs, **wkwargs)
    return None


# routine to pick a default sklearn model
all_models = [None, "K Nearest Neighbors", "Random Forest", "Gradient Boosting"]

# main display space
st.header("Multiclass Probability Visualizer - Welcome!")
# st.info("Here is why this thing is useful.")

# side bar controls: data, model, plot aesthetics
with st.sidebar:
    # data (only toy data sets for now)
    st.subheader(
        "Data Set",
        help="Pick a data set and two of its numerical features (columns) \
        \nthat will be used for model trainig. Currently, only two \
        \n'toy' data sets are available: wine and iris.\
        \n(https://scikit-learn.org/stable/datasets/toy_dataset.html)"
    )
    if st.checkbox("Synthetic Data Set", False, disabled=True):
        pass

    if st.checkbox("Toy Data Set", True, disabled=True):
        set_name = st.selectbox(
            "Select one of the Toy Data Sets", [None, "Wine", "Iris"]
        )

        # once set is chosen, process data and offer to pick X and Y features
        if set_name is not None:
            data, target = process_toy(set_name)
            st.write("Pick Features:")
            f1 = st.selectbox("Pick Feature X", data.columns)
            f2 = st.selectbox("Pick Feature Y",
                              data.columns[data.columns != f1])

    st.subheader("Classifier Model")
    # FIXME disable model selection and set to None when data set is not picked
    model_pick = st.radio("Select one of the Classifiers", all_models)
    model = fetch_model(model_pick)

    # set `random_state` if it is relevant
    if (model is not None) and (set_name is not None):
        hp = {}
        hp_desc = parse_param_desc(model)
        st.info("\n".join(model.__doc__ .split("\n\n")[:2]))
        if "random_state" in model.get_params().keys():
            hp["random_state"] = st.number_input(
                "Input Random State", 0, 500, 1, 1, help=hp_desc["random_state"])

        if isinstance(model, KNeighborsClassifier):
            hp["n_neighbors"] = st.slider("N Neighbors", 1, 100, 5)
            hp["p"] = st.slider("Power", 1, 100, 2)

        if isinstance(model, RandomForestClassifier):
            hp["n_estimators"] = st.slider(
                'Number of Estimators', 1, 500, 100, help=hp_desc["n_estimators"])
            hp["criterion"] = st.selectbox(
                'Criterion', ['gini', 'entropy'], help=hp_desc["criterion"])
            hp["max_depth"] = none_or_widget(
                'max_depth', 1, 20, 5, help=hp_desc["max_depth"])
            hp["min_samples_split"] = st.slider(
                'Min Samples Split', 2, 20, 2, help=hp_desc["min_samples_split"])
            hp["min_samples_leaf"] = st.slider(
                'Min Samples Leaf', 1, 20, 1, help=hp_desc["min_samples_leaf"])
            hp["min_weight_fraction_leaf"] = st.number_input(
                'Min Weight Fraction Leaf', 0.0, 0.5, 0.0, 0.01, help=hp_desc["min_weight_fraction_leaf"]
            )
            hp["max_features"] = st.selectbox(
                'Max Features', ['sqrt', 'log2', None], help=hp_desc["max_features"])
            hp["max_leaf_nodes"] = none_or_widget(
                'max_leaf_nodes', 2, 100, help=hp_desc["max_leaf_nodes"])
            hp["min_impurity_decrease"] = st.number_input(
                'Min Impurity Decrease', 0.0, 1.0, 0.0, 0.01, help=hp_desc["min_impurity_decrease"]
            )
            hp["bootstrap"] = st.checkbox(
                'Bootstrap', True, help=hp_desc["bootstrap"])
            if hp["bootstrap"]:
                hp["oob_score"] = st.checkbox(
                    'OOB score', False, help=hp_desc["oob_score"])
            else:
                hp["oob_score"] = False
            hp["class_weight"] = st.selectbox(
                'Class Weight', [None, 'balanced', 'balanced_subsample'], help=hp_desc["class_weight"])
            hp["ccp_alpha"] = st.number_input(
                'CCP Alpha', min_value=0.0, value=0.0, step=0.01, help=hp_desc["ccp_alpha"]
            )
            if data is not None:
                hp["max_samples"] = none_or_widget(
                    "max_samples", 1, data.shape[0], 5, help=hp_desc["max_samples"])

        if isinstance(model, GradientBoostingClassifier):
            if target.nunique() == 2:
                hp["loss"] = st.selectbox(
                    "loss", ['log_loss', 'exponential'], help=hp_desc["loss"])
            else:
                hp["loss"] = st.selectbox(
                    "loss", ['log_loss'], help=hp_desc["loss"])
            hp['learning_rate'] = st.number_input('Learning Rate', 0.0, 1.0, 0.1, 0.01, help=hp_desc["learning_rate"])
            hp['n_estimators'] = st.slider('Number of Estimators', 1, 500, 100, help=hp_desc["n_estimators"])
            hp['subsample'] = st.number_input('Subsample', 0.01, 1.0, 1.0, 0.01, help=hp_desc["subsample"])
            hp['criterion'] = st.selectbox('Criterion', ['friedman_mse', 'squared_error'], 0, help=hp_desc["criterion"])
            hp['min_samples_split'] = st.slider('Min Samples Split', 2, 500, 2, help=hp_desc["min_samples_split"])
            hp['min_samples_leaf'] = st.slider('Min Samples Leaf', 1, 500, 1, help=hp_desc["min_samples_leaf"])
            hp['min_weight_fraction_leaf'] = st.number_input('Min Weight Fraction Leaf', 0.0, 0.5, 0.0, 0.01, help=hp_desc["min_weight_fraction_leaf"])
            hp['max_depth'] = st.slider('Max Depth', 1, 500, 3, help=hp_desc["max_depth"])
            hp['min_impurity_decrease'] = st.number_input('Min Impurity Decrease', 0.0, 1.0, 0.0, 0.01, help=hp_desc["min_impurity_decrease"])
            hp['init'] = none_or_widget("Init", ["zero"], widget=st.selectbox, help=hp_desc["init"])
            hp['max_features'] = none_or_widget(
                "max_features", ['sqrt', 'log2'], widget=st.selectbox, help=hp_desc["max_features"])
            hp['max_leaf_nodes'] = none_or_widget(
                "max_leaf_nodes", 2, 500, 10, 1, help=hp_desc["max_leaf_nodes"]
            )
            hp['validation_fraction'] = st.number_input('Validation Fraction', 0.01, 0.99, 0.1, 0.01, help=hp_desc["validation_fraction"])
            hp['n_iter_no_change'] = none_or_widget(
                "n_iter_no_change", 1, 500, 10, 1, help=hp_desc["n_iter_no_change"]
            )
            hp['tol'] = st.number_input('Tol', 0., 1., 1e-4, 1e-4, help=hp_desc["tol"])
            hp['ccp_alpha'] = st.number_input('CCP Alpha', 0.0, 1.0, 0.0, 0.01, help=hp_desc["ccp_alpha"])

# If data is None, don't plot anything
# If data is not None but model is None, plot blank scatter
# if data and model are not None, plot contour
if set_name is not None:
    if 'p_v' not in st.session_state:
        st.session_state['p_v'] = ProbaVis(model, data, target, [f1, f2])
    else:
        # XXX this is quite expensive method, ought to avoid when no change in data input
        st.session_state['p_v'].set_data(data, target, [f1, f2])
    if model is None:
        st.pyplot(
            st.session_state['p_v'].plot(
                contour_on=False, return_fig=True, fig_size=(16, 9)
            )
        )
    else:
        st.session_state['p_v'].set_model(model.set_params(**hp))
        st.pyplot(
            st.session_state['p_v'].plot(
                contour_on=True, return_fig=True, fig_size=(16, 9)
            )
        )
