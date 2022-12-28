from ConfigSpace import ConfigurationSpace
from ConfigSpace import UniformFloatHyperparameter, UniformIntegerHyperparameter
from ConfigSpace.hyperparameters import CategoricalHyperparameter, UnParametrizedHyperparameter
from sklearn.ensemble import RandomForestClassifier
# from lightgbm import LGBMClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier


class ParamSpace:
    def __init__(self):

        # The machine learning algorithm
        self.ml_algorithm = None

        # A JSON-valid name to identify the algorithm
        self.name = None

        # The configuration space for the machine learning algorithm
        self.configuration_space = None

        # Whether the algorithm is deterministic
        self.is_deterministic = None

    def initialize_algorithm(self, random_state: int = None, **configuration):
        """

        Parameters
        ----------
        random_state: int
            A seed for reproducible runs
        configuration: keyword arguments
            The arguments being passed to initiate the machine learning algorithm

        Returns
        -------
        ml_algorithm: Any machine learning algorithm
            The initialized algorithm
        """

        # Some simple transformations
        config = {k: configuration[k] for k in configuration}
        config = {k: True if i == "True" else False if i == "False" else i for k, i in config.items()}
        config = {k: None if i == "None" else i for k, i in config.items()}

        # Create model
        ml_algorithm = self._initialize_algorithm(random_state=random_state, **config)

        return ml_algorithm

    def _initialize_algorithm(self, random_state=None, **config):
        try:
            return self.ml_algorithm(random_state=random_state, **config)
        except:
            pass
        return self.ml_algorithm(**config)


class RandomForestSpace(ParamSpace):
    def __init__(self):
        super().__init__()
        self.name = "RandomForest"
        self.model = RandomForestClassifier
        self.is_deterministic = False
        cs = ConfigurationSpace()
        criterion = CategoricalHyperparameter(
            "criterion", ["gini", "entropy"], default_value="gini"
        )

        # The maximum number of features used in the forest is calculated as
        # m^max_features, where m is the total number of features, and max_features
        # is the hyperparameter specified below. The default is 0.5, which yields
        # sqrt(m) features as max_features in the estimator.
        # This corresponds with Geurts' heuristic.
        max_features = UniformFloatHyperparameter(
            "max_features", 0.0, 1.0, default_value=0.5
        )

        max_depth = UnParametrizedHyperparameter("max_depth", "None")
        min_samples_split = UniformIntegerHyperparameter(
            "min_samples_split", 2, 20, default_value=2
        )
        min_samples_leaf = UniformIntegerHyperparameter(
            "min_samples_leaf", 1, 20, default_value=1
        )
        min_weight_fraction_leaf = UnParametrizedHyperparameter(
            "min_weight_fraction_leaf", 0.0
        )
        max_leaf_nodes = UnParametrizedHyperparameter("max_leaf_nodes", "None")
        min_impurity_decrease = UnParametrizedHyperparameter(
            "min_impurity_decrease", 0.0
        )
        bootstrap = CategoricalHyperparameter(
            "bootstrap", ["True", "False"], default_value="True"
        )
        cs.add_hyperparameters(
            [
                criterion,
                max_features,
                max_depth,
                min_samples_split,
                min_samples_leaf,
                min_weight_fraction_leaf,
                max_leaf_nodes,
                bootstrap,
                min_impurity_decrease,
            ]
        )

        self.configuration_space = cs
        # self.configuration_space = ConfigurationSpace()
        # self.configuration_space.add_hyperparameters([
        #     UniformFloatHyperparameter("colsample_bytree", 0.20, 0.80, default_value=0.70),
        #     UniformFloatHyperparameter("subsample", 0.20, 0.80, default_value=0.66),
        #     UniformIntegerHyperparameter("num_leaves", 4, 64, default_value=32),
        #     UniformIntegerHyperparameter("min_child_samples", 1, 100, default_value=20),
        #     UniformIntegerHyperparameter("max_depth", 4, 12, default_value=12),
        # ])

    def _initialize_algorithm(self, random_state=None, **config):
        return self.model(n_estimators=1, verbose=0, n_jobs=-1,
                          random_state=random_state, **config)


# class GradientBoostingSpace(ParamSpace):
#     def __init__(self):
#         super().__init__()
#         self.name = "GBM"
#         self.model = LGBMClassifier
#         self.is_deterministic = True
#         self.configuration_space = ConfigurationSpace()
#         self.configuration_space.add_hyperparameters([
#             UniformIntegerHyperparameter("num_leaves", 4, 64, default_value=32),
#             UniformIntegerHyperparameter("min_child_samples", 1, 100, default_value=20),
#             UniformIntegerHyperparameter("max_depth", 3, 12, default_value=12),
#             UniformFloatHyperparameter("reg_alpha", 0, 1, default_value=0),
#             UniformFloatHyperparameter("reg_lambda", 0, 1, default_value=0),
#             CategoricalHyperparameter('boosting_type', choices=["gbdt", "dart", "goss"])
#         ])
#
#     def _initialize_algorithm(self, random_state=None, **config):
#         return self.model(n_estimators=100, verbose=-1, n_jobs=-1, random_state=random_state, **config)


class DecisionTreeSpace(ParamSpace):
    def __init__(self):
        super().__init__()
        self.name = "DecisionTree"
        self.model = DecisionTreeClassifier
        self.is_deterministic = False
        self.configuration_space = ConfigurationSpace()
        self.configuration_space.add_hyperparameters([
            CategoricalHyperparameter("criterion", ["gini", "entropy"], default_value="gini"),
            UniformIntegerHyperparameter('max_depth', 1, 20, default_value=20),
            UniformIntegerHyperparameter("min_samples_split", 2, 20, default_value=2),
            UniformIntegerHyperparameter("min_samples_leaf", 1, 20, default_value=1)
        ])

    def _initialize_algorithm(self, random_state=None, **config):
        return self.model(min_weight_fraction_leaf=0, max_features=1.0, max_leaf_nodes=None,
                          min_impurity_decrease=0.0, random_state=random_state,
                          **config)


class SVMSpace(ParamSpace):
    def __init__(self):
        super().__init__()
        self.name = "SVM"
        self.model = LinearSVC
        self.is_deterministic = False
        self.configuration_space = ConfigurationSpace()
        self.configuration_space.add_hyperparameters([
            UniformFloatHyperparameter("tol", 1e-5, 1e-1, default_value=1e-4, log=True),
            UniformFloatHyperparameter("C", 0.03125, 32768, log=True, default_value=1.0)
        ])

    def _initialize_algorithm(self, random_state=None, **config):
        return self.model(penalty="l2", loss="squared_hinge", dual=False, multi_class="ovr", fit_intercept=True,
                          intercept_scaling=1, **config)
