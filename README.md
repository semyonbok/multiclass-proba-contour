# ProbaVis
Inspired by [numerous examples](https://scikit-learn.org/stable/auto_examples/index.html) given in scikit-learn documentation and the [helper module](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/inspection/_plot/decision_boundary.py) provided in a [fantastic MOOC](https://www.fun-mooc.fr/en/courses/machine-learning-python-scikit-learn/), the implemented module enables the visualisation of predicted class probabilities for a data set with more than two classes. For instance, the figures below illustrate a synthesised data set containing samples with four classes; the predicted probability contours are obtained with `sklearn.neighbors.KNeighborsClassifier` and `sklearn.ensemble.RandomForestClassifier` trained on the two numerical features. In addition, `replot` method can be passed to `ipywidgets.interact` function in a Jupyter IDLE, allowing to adjust the model's hyperparameters and immediately observe the changes in contour. 
## Training Data Set
![image](https://user-images.githubusercontent.com/94805866/166163074-6eb26a9d-d6c6-4c7d-860a-1bf9d9e1c5b7.png)

## K Nearest Neigbors
![image](https://user-images.githubusercontent.com/94805866/166163537-976b8c0d-911d-4fa9-8571-5b625a734a8d.png)

## Random Forest
![image](https://user-images.githubusercontent.com/94805866/166163493-3c123e4a-2a98-4922-8a97-4122d0d02d0d.png)

## Future Work
- [X] Edit documentation and type hinting,
- [ ] Allow for a contour plot customisation,
- [ ] Implement a GUI with streamlit.
