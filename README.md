# ProbaVis
Inspired by <a href="https://scikit-learn.org/stable/auto_examples/index.html" >numerous examples</a> given in scikit-learn documentation, the implemented Python module expands the functionality of the <a href="https://github.com/INRIA/scikit-learn-mooc/blob/main/python_scripts/helpers/plotting.py" >helper module</a> provided in a <a href="https://www.fun-mooc.fr/en/courses/machine-learning-python-scikit-learn/" >fantastic MOOC</a> by enabling the visualisation of predicted class probabilities for a data set with more than two classes. In addition, `replot` method can be passed to `ipywidgets.interact` function in a Jupyter IDLE, allowing to interactively change the model's hyperparameters and immediately observe the changes in contour. The figures below illustrate the synthesised data set with two numerical features and predicted probability contours obtained with `sklearn.neighbors.KNeighborsClassifier` and `sklearn.ensemble.RandomForestClassifier`. 
## Training Data Set
![image](https://user-images.githubusercontent.com/94805866/166163074-6eb26a9d-d6c6-4c7d-860a-1bf9d9e1c5b7.png)

## K Nearest Neigbors
![image](https://user-images.githubusercontent.com/94805866/166163537-976b8c0d-911d-4fa9-8571-5b625a734a8d.png)

## Random Forest
![image](https://user-images.githubusercontent.com/94805866/166163493-3c123e4a-2a98-4922-8a97-4122d0d02d0d.png)

## Future Work
- [ ] Edit documentation,
- [ ] Allow for a contour plot customisation,
- [ ] Implement a GUI with streamlit.
