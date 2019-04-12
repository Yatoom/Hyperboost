# Random Forest based on LightGBM
from ConfigSpace import ConfigurationSpace
from ConfigSpace import UniformFloatHyperparameter, UniformIntegerHyperparameter
from ConfigSpace.conditions import EqualsCondition, InCondition
from ConfigSpace.forbidden import ForbiddenAndConjunction, ForbiddenEqualsClause, SingleValueForbiddenClause
from ConfigSpace.hyperparameters import CategoricalHyperparameter, Constant, UnParametrizedHyperparameter
from lightgbm import LGBMClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


class RandomForestSpace:
    # Properties
    model = LGBMClassifier
    is_deterministic = False
    name = "RandomForest"

    # Hyper parameter space
    cs = ConfigurationSpace()
    num_leaves = UniformIntegerHyperparameter("num_leaves", 4, 64, default_value=32)
    cs.add_hyperparameters([
        UniformFloatHyperparameter("colsample_bytree", 0.20, 0.80, default_value=0.70),
        UniformFloatHyperparameter("subsample", 0.20, 0.80, default_value=0.66),
        num_leaves,
        UniformIntegerHyperparameter("min_child_samples", 1, 100, default_value=20),
        UniformIntegerHyperparameter("max_depth", 4, 12, default_value=12),
        UniformFloatHyperparameter("reg_alpha", 0, 1, default_value=0),
        UniformFloatHyperparameter("reg_lambda", 0, 1, default_value=0),
        Constant("n_estimators", 100),
        Constant("subsample_freq", 1),
        Constant("boosting_type", "rf"),
        Constant("verbose", -1),
        Constant("n_jobs", 1),
    ])
    # cs.add_condition(SingleValueForbiddenClause(num_leaves, 31))


class DecisionTreeSpace:
    # Properties
    model = DecisionTreeClassifier
    is_deterministic = True
    name = "DecisionTree"

    # Hyper parameter space
    cs = ConfigurationSpace()

    criterion = CategoricalHyperparameter("criterion", ["gini", "entropy"], default_value="gini")
    max_depth = UniformIntegerHyperparameter('max_depth', 1, 20, default_value=20)
    min_samples_split = UniformIntegerHyperparameter("min_samples_split", 2, 20, default_value=2)
    min_samples_leaf = UniformIntegerHyperparameter("min_samples_leaf", 1, 20, default_value=1)
    min_weight_fraction_leaf = Constant("min_weight_fraction_leaf", 0.0)
    max_features = UnParametrizedHyperparameter('max_features', 1.0)
    max_leaf_nodes = UnParametrizedHyperparameter("max_leaf_nodes", "None")
    min_impurity_decrease = UnParametrizedHyperparameter('min_impurity_decrease', 0.0)

    cs.add_hyperparameters([criterion, max_features, max_depth,
                            min_samples_split, min_samples_leaf,
                            min_weight_fraction_leaf, max_leaf_nodes,
                            min_impurity_decrease])


class LDASpace:
    # Properties
    model = LinearDiscriminantAnalysis
    is_deterministic = True
    name = "LDA"

    # Hyper parameter space
    cs = ConfigurationSpace()
    shrinkage = CategoricalHyperparameter("shrinkage", ["None", "auto", "manual"], default_value="None")
    shrinkage_factor = UniformFloatHyperparameter("shrinkage_factor", 0., 1., 0.5)
    n_components = UniformIntegerHyperparameter('n_components', 1, 250, default_value=10)
    tol = UniformFloatHyperparameter("tol", 1e-5, 1e-1, default_value=1e-4, log=True)
    cs.add_hyperparameters([shrinkage, shrinkage_factor, n_components, tol])

    cs.add_condition(EqualsCondition(shrinkage_factor, shrinkage, "manual"))


class AdaboostSpace:
    # Properties
    model = AdaBoostClassifier
    is_deterministic = True
    name = "Adaboost"

    # Hyper parameter space
    cs = ConfigurationSpace()

    n_estimators = UniformIntegerHyperparameter(name="n_estimators", lower=50, upper=500, default_value=50, log=False)
    learning_rate = UniformFloatHyperparameter(name="learning_rate", lower=0.01, upper=2, default_value=0.1, log=True)
    algorithm = CategoricalHyperparameter(name="algorithm", choices=["SAMME.R", "SAMME"], default_value="SAMME.R")
    max_depth = UniformIntegerHyperparameter(name="max_depth", lower=1, upper=10, default_value=1, log=False)
    cs.add_hyperparameters([n_estimators, learning_rate, algorithm, max_depth])


class ExtraTreesSpace:
    # Properties
    model = ExtraTreesClassifier
    is_deterministic = True
    name = "ExtraTrees"

    # Hyperparameter space
    cs = ConfigurationSpace()

    n_estimators = Constant("n_estimators", 100)
    criterion = CategoricalHyperparameter(
        "criterion", ["gini", "entropy"], default_value="gini")

    # The maximum number of features used in the forest is calculated as m^max_features, where
    # m is the total number of features, and max_features is the hyperparameter specified below.
    # The default is 0.5, which yields sqrt(m) features as max_features in the estimator. This
    # corresponds with Geurts' heuristic.
    max_features = UniformFloatHyperparameter(
        "max_features", 0., 1., default_value=0.5)

    max_depth = UnParametrizedHyperparameter(name="max_depth", value="None")

    min_samples_split = UniformIntegerHyperparameter(
        "min_samples_split", 2, 20, default_value=2)
    min_samples_leaf = UniformIntegerHyperparameter(
        "min_samples_leaf", 1, 20, default_value=1)
    min_weight_fraction_leaf = UnParametrizedHyperparameter('min_weight_fraction_leaf', 0.)
    max_leaf_nodes = UnParametrizedHyperparameter("max_leaf_nodes", "None")
    min_impurity_decrease = UnParametrizedHyperparameter('min_impurity_decrease', 0.0)

    bootstrap = CategoricalHyperparameter(
        "bootstrap", ["True", "False"], default_value="False")
    n_jobs = Constant("n_jobs", 4)
    cs.add_hyperparameters([n_estimators, criterion, max_features,
                            max_depth, min_samples_split, min_samples_leaf,
                            min_weight_fraction_leaf, max_leaf_nodes,
                            min_impurity_decrease, bootstrap, n_jobs])


class SVMSpace:
    # Properties
    model = SVC
    is_deterministic = True
    name = "SVM"

    # Hyperparameter space
    C = UniformFloatHyperparameter("C", 0.03125, 32768, log=True,
                                   default_value=1.0)
    # No linear kernel here, because we have liblinear
    kernel = CategoricalHyperparameter(name="kernel",
                                       choices=["rbf", "poly", "sigmoid"],
                                       default_value="rbf")
    degree = UniformIntegerHyperparameter("degree", 2, 5, default_value=3)
    gamma = UniformFloatHyperparameter("gamma", 3.0517578125e-05, 8,
                                       log=True, default_value=0.1)
    # TODO this is totally ad-hoc
    coef0 = UniformFloatHyperparameter("coef0", -1, 1, default_value=0)
    # probability is no hyperparameter, but an argument to the SVM algo
    shrinking = CategoricalHyperparameter("shrinking", ["True", "False"],
                                          default_value="True")
    tol = UniformFloatHyperparameter("tol", 1e-5, 1e-1, default_value=1e-3,
                                     log=True)
    # cache size is not a hyperparameter, but an argument to the program!
    max_iter = UnParametrizedHyperparameter("max_iter", -1)

    cs = ConfigurationSpace()
    cs.add_hyperparameters([C, kernel, degree, gamma, coef0, shrinking,
                            tol, max_iter])

    degree_depends_on_poly = EqualsCondition(degree, kernel, "poly")
    coef0_condition = InCondition(coef0, kernel, ["poly", "sigmoid"])
    cs.add_condition(degree_depends_on_poly)
    cs.add_condition(coef0_condition)
