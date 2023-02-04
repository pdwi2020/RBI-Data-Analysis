# RBI-Data-Analysis
# An AdaBoost classifier.

An AdaBoost [1] classifier is a meta-estimator that begins by fitting a classifier on the original dataset and then fits additional copies of the classifier on the same dataset but where the weights of incorrectly classified instances are adjusted such that subsequent classifiers focus more on difficult cases.

This class implements the algorithm known as AdaBoost-SAMME [2].

Read more in the User Guide.

New in version 0.14.

Parameters:
estimatorobject, default=None
The base estimator from which the boosted ensemble is built. Support for sample weighting is required, as well as proper classes_ and n_classes_ attributes. If None, then the base estimator is DecisionTreeClassifier initialized with max_depth=1.

New in version 1.2: base_estimator was renamed to estimator.

n_estimatorsint, default=50
The maximum number of estimators at which boosting is terminated. In case of perfect fit, the learning procedure is stopped early. Values must be in the range [1, inf).

learning_ratefloat, default=1.0
Weight applied to each classifier at each boosting iteration. A higher learning rate increases the contribution of each classifier. There is a trade-off between the learning_rate and n_estimators parameters. Values must be in the range (0.0, inf).

algorithm{‘SAMME’, ‘SAMME.R’}, default=’SAMME.R’
If ‘SAMME.R’ then use the SAMME.R real boosting algorithm. estimator must support calculation of class probabilities. If ‘SAMME’ then use the SAMME discrete boosting algorithm. The SAMME.R algorithm typically converges faster than SAMME, achieving a lower test error with fewer boosting iterations.

random_stateint, RandomState instance or None, default=None
Controls the random seed given at each estimator at each boosting iteration. Thus, it is only used when estimator exposes a random_state. Pass an int for reproducible output across multiple function calls. See Glossary.

base_estimatorobject, default=None
The base estimator from which the boosted ensemble is built. Support for sample weighting is required, as well as proper classes_ and n_classes_ attributes. If None, then the base estimator is DecisionTreeClassifier initialized with max_depth=1.

Deprecated since version 1.2: base_estimator is deprecated and will be removed in 1.4. Use estimator instead.
Attributes:
estimator_estimator
Estimator used to grow the ensemble.

base_estimator_estimator
Estimator used to grow the ensemble.

estimators_list of classifiers
The collection of fitted sub-estimators.

classes_ndarray of shape (n_classes,)
The classes labels.

n_classes_int
The number of classes.

estimator_weights_ndarray of floats
Weights for each estimator in the boosted ensemble.

estimator_errors_ndarray of floats
Classification error for each estimator in the boosted ensemble.

feature_importances_ndarray of shape (n_features,)
The impurity-based feature importances.

n_features_in_int
Number of features seen during fit.

New in version 0.24.

feature_names_in_ndarray of shape (n_features_in_,)
Names of features seen during fit. Defined only when X has feature names that are all strings.

New in version 1.0.

See also
AdaBoostRegressor
An AdaBoost regressor that begins by fitting a regressor on the original dataset and then fits additional copies of the regressor on the same dataset but where the weights of instances are adjusted according to the error of the current prediction.

GradientBoostingClassifier
GB builds an additive model in a forward stage-wise fashion. Regression trees are fit on the negative gradient of the binomial or multinomial deviance loss function. Binary classification is a special case where only a single regression tree is induced.

sklearn.tree.DecisionTreeClassifier
A non-parametric supervised learning method used for classification. Creates a model that predicts the value of a target variable by learning simple decision rules inferred from the data features.

References

[1]
Y. Freund, R. Schapire, “A Decision-Theoretic Generalization of on-Line Learning and an Application to Boosting”, 1995.

[2]
Zhu, H. Zou, S. Rosset, T. Hastie, “Multi-class AdaBoost”, 2009.

Examples

>>>
>>> from sklearn.ensemble import AdaBoostClassifier
>>> from sklearn.datasets import make_classification
>>> X, y = make_classification(n_samples=1000, n_features=4,
...                            n_informative=2, n_redundant=0,
...                            random_state=0, shuffle=False)
>>> clf = AdaBoostClassifier(n_estimators=100, random_state=0)
>>> clf.fit(X, y)
AdaBoostClassifier(n_estimators=100, random_state=0)
>>> clf.predict([[0, 0, 0, 0]])
array([1])
>>> clf.score(X, y)
0.983...
