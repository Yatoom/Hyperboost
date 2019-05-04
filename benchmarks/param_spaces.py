# Random Forest based on LightGBM
from ConfigSpace import ConfigurationSpace
from ConfigSpace import UniformFloatHyperparameter, UniformIntegerHyperparameter
from ConfigSpace.hyperparameters import CategoricalHyperparameter, Constant, UnParametrizedHyperparameter
from lightgbm import LGBMClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier, AdaBoostClassifier
from sklearn.svm import LinearSVC
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
        # UniformFloatHyperparameter("reg_alpha", 0, 1, default_value=0),
        # UniformFloatHyperparameter("reg_lambda", 0, 1, default_value=0),
    ])

    @staticmethod
    def from_cfg(random_state=None, **cfg):
        return RandomForestSpace.model(n_estimators=100, subsample_freq=1, boosting_type="rf", verbose=-1, n_jobs=-1,
                                       random_state=random_state, **cfg)


class DecisionTreeSpace:
    # Properties
    model = DecisionTreeClassifier
    is_deterministic = False
    name = "DecisionTree-stoch"

    # Hyper parameter space
    cs = ConfigurationSpace()

    cs.add_hyperparameters([
        CategoricalHyperparameter("criterion", ["gini", "entropy"], default_value="gini"),
        UniformIntegerHyperparameter('max_depth', 1, 20, default_value=20),
        UniformIntegerHyperparameter("min_samples_split", 2, 20, default_value=2),
        UniformIntegerHyperparameter("min_samples_leaf", 1, 20, default_value=1)
    ])

    @staticmethod
    def from_cfg(random_state=None, **cfg):
        return DecisionTreeSpace.model(min_weight_fraction_leaf=0, max_features=1.0, max_leaf_nodes=None,
                                       min_impurity_decrease=0.0, random_state=random_state,
                                       **cfg)


class LDASpace:
    # Properties
    model = LinearDiscriminantAnalysis
    is_deterministic = True
    name = "LDA"

    # Hyper parameter space
    cs = ConfigurationSpace()
    # shrinkage = CategoricalHyperparameter("shrinkage", ["None", "auto", "manual"], default_value="None")
    shrinkage = UniformFloatHyperparameter("shrinkage", 0., 1., 0.5)
    n_components = UniformIntegerHyperparameter('n_components', 1, 250, default_value=10)
    tol = UniformFloatHyperparameter("tol", 1e-5, 1e-1, default_value=1e-4, log=True)
    # solver = CategoricalHyperparameter("solver", ["lsqr", "eigen"], default_value="eigen")
    cs.add_hyperparameters([shrinkage, n_components, tol])

    # cs.add_condition(EqualsCondition(shrinkage_factor, shrinkage, "manual"))


class AdaboostSpace:
    # Properties
    model = AdaBoostClassifier
    is_deterministic = False
    name = "Adaboost"

    # Hyper parameter space
    cs = ConfigurationSpace()

    n_estimators = UniformIntegerHyperparameter(name="n_estimators", lower=50, upper=500, default_value=50, log=False)
    learning_rate = UniformFloatHyperparameter(name="learning_rate", lower=0.01, upper=2, default_value=0.1, log=True)
    algorithm = CategoricalHyperparameter(name="algorithm", choices=["SAMME.R", "SAMME"], default_value="SAMME.R")
    max_depth = UniformIntegerHyperparameter(name="max_depth", lower=1, upper=10, default_value=1, log=False)
    cs.add_hyperparameters([n_estimators, learning_rate, algorithm, max_depth])

    @staticmethod
    def from_cfg(random_state=None, **cfg):
        max_depth = cfg['max_depth']
        n_estimators = cfg['n_estimators']
        learning_rate = cfg['learning_rate']
        algorithm = cfg['algorithm']
        return AdaboostSpace.model(n_estimators=n_estimators, learning_rate=learning_rate,
                                   algorithm=algorithm, base_estimator=DecisionTreeClassifier(max_depth=max_depth),
                                   random_state=random_state)


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
    n_jobs = Constant("n_jobs", -1)
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

    tol = UniformFloatHyperparameter(
        "tol", 1e-5, 1e-1, default_value=1e-4, log=True)
    C = UniformFloatHyperparameter(
        "C", 0.03125, 32768, log=True, default_value=1.0)

    cs.add_hyperparameters([tol, C])

    @staticmethod
    def from_cfg(random_state=None, **cfg):
        return SVMSpace.model(penalty="l2", loss="squared_hing", dual=False, multi_class="ovr", fit_intercept=True,
                              intercept_scaling=1, **cfg)
