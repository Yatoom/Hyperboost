# Random Forest based on LightGBM
from ConfigSpace import ConfigurationSpace
from ConfigSpace import UniformFloatHyperparameter, UniformIntegerHyperparameter
from ConfigSpace.conditions import EqualsCondition, InCondition
from ConfigSpace.forbidden import ForbiddenAndConjunction, ForbiddenEqualsClause, SingleValueForbiddenClause
from ConfigSpace.hyperparameters import CategoricalHyperparameter, Constant, UnParametrizedHyperparameter
from lightgbm import LGBMClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier


class RandomForestSpace:
    # Properties
    model = LGBMClassifier
    is_deterministic = False
    name = "RandomForest"

    # Hyper parameter space
    cs = ConfigurationSpace()
    cs.add_hyperparameters([
        UniformFloatHyperparameter("colsample_bytree", 0.20, 0.80, default_value=0.70),
        UniformFloatHyperparameter("subsample", 0.20, 0.80, default_value=0.66),
        UniformIntegerHyperparameter("num_leaves", 4, 64, default_value=32),
        UniformIntegerHyperparameter("min_child_samples", 1, 100, default_value=20),
        UniformIntegerHyperparameter("max_depth", 4, 12, default_value=12),
        UniformFloatHyperparameter("reg_alpha", 0, 1, default_value=0),
        UniformFloatHyperparameter("reg_lambda", 0, 1, default_value=0),
        Constant("n_estimators", 100),
        Constant("subsample_freq", 1),
        Constant("boosting_type", "rf"),
        Constant("verbose", -1),
        Constant("n_jobs", -1),
    ])
    # cs.add_condition(SingleValueForbiddenClause(num_leaves, 31))


class DecisionTreeSpace:
    # Properties
    model = DecisionTreeClassifier
    is_deterministic = False
    name = "DecisionTree-stoch"

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
    model = LinearSVC
    is_deterministic = True
    name = "SVM"

    # Hyperparameter space
    cs = ConfigurationSpace()

    # penalty = CategoricalHyperparameter(
    #     "penalty", ["l1", "l2"], default_value="l2")
    penalty = Constant("penalty", "l2")
    # loss = CategoricalHyperparameter(
    #     "loss", ["hinge", "squared_hinge"], default_value="squared_hinge")
    loss = Constant("loss", "squared_hinge")
    dual = Constant("dual", "False")
    # This is set ad-hoc
    tol = UniformFloatHyperparameter(
        "tol", 1e-5, 1e-1, default_value=1e-4, log=True)
    C = UniformFloatHyperparameter(
        "C", 0.03125, 32768, log=True, default_value=1.0)
    multi_class = Constant("multi_class", "ovr")
    # These are set ad-hoc
    fit_intercept = Constant("fit_intercept", "True")
    intercept_scaling = Constant("intercept_scaling", 1)
    cs.add_hyperparameters([penalty, loss, dual, tol, C, multi_class,
                            fit_intercept, intercept_scaling])

    # penalty_and_loss = ForbiddenAndConjunction(
    #     ForbiddenEqualsClause(penalty, "l1"),
    #     ForbiddenEqualsClause(loss, "hinge")
    # )
    # constant_penalty_and_loss = ForbiddenAndConjunction(
    #     ForbiddenEqualsClause(dual, "False"),
    #     ForbiddenEqualsClause(penalty, "l2"),
    #     ForbiddenEqualsClause(loss, "hinge")
    # )
    # penalty_and_dual = ForbiddenAndConjunction(
    #     ForbiddenEqualsClause(dual, "False"),
    #     ForbiddenEqualsClause(penalty, "l1")
    # )
    # cs.add_forbidden_clause(penalty_and_loss)
    # cs.add_forbidden_clause(constant_penalty_and_loss)
    # cs.add_forbidden_clause(penalty_and_dual)