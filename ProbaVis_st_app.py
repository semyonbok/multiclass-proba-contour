import streamlit as st
from sklearn.datasets import load_iris, load_wine
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import multiclass_proba_contour as mpc


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


# routine to pick a default sklearn model
all_models = [m()
              for m in [RandomForestClassifier, GradientBoostingClassifier]]

# main display space
st.header("Multiclass Probability Visualizer - Welcome!")
st.info("Here is why this thing is useful.")

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
    model = st.selectbox("Select one of the Classifiers", [None] + all_models)
    hp = {}

    # set `random_state` if it is relevant
    if model is not None:
        st.info(model.__doc__ .split("\n\n")[1])
        if "random_state" in model.get_params().keys():
            hp["random_state"] = st.number_input(
                "Input Random State", value=0, step=1)

    if isinstance(model, RandomForestClassifier):
        hp["criterion"] = st.radio("criterion", ["gini", "entropy"])
        hp["n_estimators"] = st.slider("n_estimators", 1, 500, 100)
        hp["max_depth"] = st.slider("max_depth", 1, 25, 5)
        hp["min_samples_leaf"] = st.slider("min_samples_leaf", 1, 25, 1)
        hp["min_impurity_decrease"] = st.slider(
            "min_impurity_decrease", 0.0, 0.2, 0.0, 0.01)

    elif isinstance(model, GradientBoostingClassifier):
        hp["loss"] = st.radio("loss", ['log_loss', 'deviance', 'exponential'])
        hp["learning_rate"] = st.slider("learning_rate", 0.01, 0.2, 0.1, 0.01)
        hp["n_estimators"] = st.slider("n_estimators", 1, 500, 100)
        hp["subsample"] = st.slider("subsample", 0.01, 1.0, 1.0, 0.01)
        hp["criterion"] = st.radio(
            "criterion", ['friedman_mse', 'squared_error'])

# If data is None, don't plot anything
# If data is not None but model is None, plot blank scatter
# if data and model are not None, plot contour
if set_name is not None:
    if 'p_v' not in st.session_state:
        st.session_state['p_v'] = mpc.ProbaVis(model, data, target, [f1, f2])
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
