import itertools
import time
from abc import ABC
from copy import copy
from typing import List, Union, Tuple, Optional

import numpy as np
from smac.configspace import (
    get_one_exchange_neighbourhood,
    Configuration,
    ConfigurationSpace,
    convert_configurations_to_array,
)
from smac.optimizer.acquisition import AbstractAcquisitionFunction
from smac.optimizer.ei_optimization import InterleavedLocalAndRandomSearch, RandomSearch, LocalSearch
from smac.runhistory.runhistory import RunHistory
from smac.stats.stats import Stats


class FasterLocalSearch(LocalSearch):
    """Implementation of SMAC's local search.

    Parameters
    ----------
    acquisition_function : ~smac.optimizer.acquisition.AbstractAcquisitionFunction

    config_space : ~smac.configspace.ConfigurationSpace

    rng : np.random.RandomState or int, optional

    max_steps: int
        Maximum number of iterations that the local search will perform

    n_steps_plateau_walk: int
        number of steps during a plateau walk before local search terminates

    vectorization_min_obtain : int
        Minimal number of neighbors to obtain at once for each local search for vectorized calls. Can be tuned to
        reduce the overhead of SMAC

    vectorization_max_obtain : int
        Maximal number of neighbors to obtain at once for each local search for vectorized calls. Can be tuned to
        reduce the overhead of SMAC

    """

    def __init__(
            self,
            acquisition_function: AbstractAcquisitionFunction,
            config_space: ConfigurationSpace,
            rng: Union[bool, np.random.RandomState] = None,
            max_steps: Optional[int] = None,
            n_steps_plateau_walk: int = 10,
            vectorization_min_obtain: int = 2,
            vectorization_max_obtain: int = 64,
    ):
        super().__init__(acquisition_function, config_space, rng)
        self.max_steps = max_steps
        self.n_steps_plateau_walk = n_steps_plateau_walk
        self.vectorization_min_obtain = vectorization_min_obtain
        self.vectorization_max_obtain = vectorization_max_obtain

    def _maximize(
            self,
            runhistory: RunHistory,
            stats: Stats,
            num_points: int,
            additional_start_points: Optional[List[Tuple[float, Configuration]]] = None,
            **kwargs
    ) -> List[Tuple[float, Configuration]]:
        """Starts a local search from the given startpoint and quits
        if either the max number of steps is reached or no neighbor
        with an higher improvement was found.

        Parameters
        ----------
        runhistory: ~smac.runhistory.runhistory.RunHistory
            runhistory object
        stats: ~smac.stats.stats.Stats
            current stats object
        num_points: int
            number of points to be sampled
        additional_start_points : Optional[List[Tuple[float, Configuration]]]
            Additional start point
        ***kwargs:
            Additional parameters that will be passed to the
            acquisition function

        Returns
        -------
        incumbent: np.array(1, D)
            The best found configuration
        acq_val_incumbent: np.array(1,1)
            The acquisition value of the incumbent

        """

        init_points = self._get_initial_points(num_points, runhistory, additional_start_points)
        configs_acq = self._do_search(init_points)

        # shuffle for random tie-break
        self.rng.shuffle(configs_acq)

        # sort according to acq value
        configs_acq.sort(reverse=True, key=lambda x: x[0])
        for _, inc in configs_acq:
            inc.origin = 'Local Search'

        return configs_acq

    def _get_initial_points(self, num_points, runhistory, additional_start_points):

        if runhistory.empty():
            init_points = self.config_space.sample_configuration(size=num_points)
        else:
            # initiate local search
            configs_previous_runs = runhistory.get_all_configs()

            # configurations with the highest previous EI
            configs_previous_runs_sorted = self._sort_configs_by_acq_value(configs_previous_runs)
            configs_previous_runs_sorted = [conf[1] for conf in configs_previous_runs_sorted[:num_points]]

            # configurations with the lowest predictive cost, check for None to make unit tests work
            if self.acquisition_function.model is not None:
                conf_array = convert_configurations_to_array(configs_previous_runs)
                costs = self.acquisition_function.model.predict_marginalized_over_instances(conf_array)[0]
                # From here
                # http://stackoverflow.com/questions/20197990/how-to-make-argsort-result-to-be-random-between-equal-values
                random = self.rng.rand(len(costs))
                # Last column is primary sort key!
                indices = np.lexsort((random.flatten(), costs.flatten()))

                # Cannot use zip here because the indices array cannot index the
                # rand_configs list, because the second is a pure python list
                configs_previous_runs_sorted_by_cost = [configs_previous_runs[ind] for ind in indices][:num_points]
            else:
                configs_previous_runs_sorted_by_cost = []

            if additional_start_points is not None:
                additional_start_points = [asp[1] for asp in additional_start_points[:num_points]]
            else:
                additional_start_points = []

            init_points = []
            init_points_as_set = set()
            for cand in itertools.chain(
                    configs_previous_runs_sorted,
                    configs_previous_runs_sorted_by_cost,
                    additional_start_points,
            ):
                if cand not in init_points_as_set:
                    init_points.append(cand)
                    init_points_as_set.add(cand)

        return init_points

    def _do_search(
            self,
            start_points: List[Configuration],
            **kwargs
    ) -> List[Tuple[float, Configuration]]:

        # Gather data strucuture for starting points
        if isinstance(start_points, Configuration):
            start_points = [start_points]
        incumbents = start_points
        # Compute the acquisition value of the incumbents
        num_incumbents = len(incumbents)
        acq_val_incumbents = self.acquisition_function(incumbents, **kwargs)
        if num_incumbents == 1:
            acq_val_incumbents = [acq_val_incumbents[0][0]]
        else:
            acq_val_incumbents = [a[0] for a in acq_val_incumbents]

        # Set up additional variables required to do vectorized local search:
        # whether the i-th local search is still running
        active = [True] * num_incumbents
        # number of plateau walks of the i-th local search. Reaching the maximum number is the stopping criterion of
        # the local search.
        n_no_plateau_walk = [0] * num_incumbents
        # tracking the number of steps for logging purposes
        local_search_steps = [0] * num_incumbents
        # tracking the number of neighbors looked at for logging purposes
        neighbors_looked_at = [0] * num_incumbents
        # tracking the number of neighbors generated for logging purposse
        neighbors_generated = [0] * num_incumbents
        # how many neighbors were obtained for the i-th local search. Important to map the individual acquisition
        # function values to the correct local search run
        obtain_n = [self.vectorization_min_obtain] * num_incumbents
        # Tracking the time it takes to compute the acquisition function
        times = []

        # Set up the neighborhood generators
        neighborhood_iterators = []
        for i, inc in enumerate(incumbents):
            neighborhood_iterators.append(get_one_exchange_neighbourhood(
                inc, seed=self.rng.randint(low=0, high=100000)))
            local_search_steps[i] += 1
        # Keeping track of configurations with equal acquisition value for plateau walking
        neighbors_w_equal_acq = [[]] * num_incumbents

        num_iters = 0
        while np.any(active):

            num_iters += 1
            # Whether the i-th local search improved. When a new neighborhood is generated, this is used to determine
            # whether a step was made (improvement) or not (iterator exhausted)
            improved = [False] * num_incumbents
            # Used to request a new neighborhood for the incumbent of the i-th local search
            new_neighborhood = [False] * num_incumbents

            # Obtain the amount of neighbors specified from active local searches
            # neigbhors = [next(n) for i, n in enumerate(neighborhood_iterators) if active[i] for _ in range(obtain_n[i])]
            neighbors_list = [
                [next(n, None) for _ in range(obtain_n[i])] if active[i] else None
                for i, n in enumerate(neighborhood_iterators)
            ]

            neighbors = []
            for i, neighbor_list in enumerate(neighbors_list):

                # Local search not active
                if neighbor_list is None:
                    continue

                for index, j in enumerate(neighbor_list):
                    if j is None:
                        new_neighborhood[i] = True
                        obtain_n[i] = index
                        break
                    neighbors_generated[i] += 1
                    neighbors.append(j)


            # neighbors = []
            # for i, neighbor_list in enumerate(neighbors_list):
            #     counter = 0
            #     if active[i]:
            #         neighbors_for_i = []
            #         for j in range(obtain_n[i]):
            #             try:
            #                 n = neighbor_list[counter]
            #                 counter += 1
            #                 neighbors_generated[i] += 1
            #                 neighbors_for_i.append(n)
            #             except IndexError:
            #                 obtain_n[i] = len(neighbors_for_i)
            #                 new_neighborhood[i] = True
            #                 break
            #         neighbors.extend(neighbors_for_i)

            if len(neighbors) != 0:
                start_time = time.time()
                acq_val = self.acquisition_function(neighbors, **kwargs)
                end_time = time.time()
                times.append(end_time - start_time)
                if np.ndim(acq_val.shape) == 0:
                    acq_val = [acq_val]

                # Comparing the acquisition function of the neighbors with the acquisition value of the incumbent
                acq_index = 0
                # Iterating the all i local searches
                for i in range(num_incumbents):
                    if not active[i]:
                        continue
                    # And for each local search we know how many neighbors we obtained
                    for j in range(obtain_n[i]):
                        # The next line is only true if there was an improvement and we basically need to iterate to
                        # the i+1-th local search
                        if improved[i]:
                            acq_index += 1
                        else:
                            neighbors_looked_at[i] += 1

                            # Found a better configuration
                            if acq_val[acq_index] > acq_val_incumbents[i]:
                                self.logger.debug(
                                    "Local search %d: Switch to one of the neighbors (after %d configurations).",
                                    i,
                                    neighbors_looked_at[i],
                                )
                                incumbents[i] = neighbors[acq_index]
                                acq_val_incumbents[i] = acq_val[acq_index]
                                new_neighborhood[i] = True
                                improved[i] = True
                                local_search_steps[i] += 1
                                neighbors_w_equal_acq[i] = []
                                obtain_n[i] = 1
                            # Found an equally well performing configuration, keeping it for plateau walking
                            elif acq_val[acq_index] == acq_val_incumbents[i]:
                                neighbors_w_equal_acq[i].append(neighbors[acq_index])

                            acq_index += 1

            # Now we check whether we need to create new neighborhoods and whether we need to increase the number of
            # plateau walks for one of the local searches. Also disables local searches if the number of plateau walks
            # is reached (and all being switched off is the termination criterion).
            for i in range(num_incumbents):
                if not active[i]:
                    continue
                if obtain_n[i] == 0 or improved[i]:
                    obtain_n[i] = 2
                else:
                    obtain_n[i] = obtain_n[i] * 2
                    obtain_n[i] = min(obtain_n[i], self.vectorization_max_obtain)
                if new_neighborhood[i]:
                    if not improved[i] and n_no_plateau_walk[i] < self.n_steps_plateau_walk:
                        if len(neighbors_w_equal_acq[i]) != 0:
                            incumbents[i] = neighbors_w_equal_acq[i][0]
                            neighbors_w_equal_acq[i] = []
                        n_no_plateau_walk[i] += 1
                    if n_no_plateau_walk[i] >= self.n_steps_plateau_walk:
                        active[i] = False
                        continue

                    neighborhood_iterators[i] = get_one_exchange_neighbourhood(
                        incumbents[i], seed=self.rng.randint(low=0, high=100000),
                    )

        self.logger.debug(
            "Local searches took %s steps and looked at %s configurations. Computing the acquisition function in "
            "vectorized for took %f seconds on average.",
            local_search_steps, neighbors_looked_at, np.mean(times),
        )

        return [(a, i) for a, i in zip(acq_val_incumbents, incumbents)]


class FasterInterleavedLocalAndRandomSearch(InterleavedLocalAndRandomSearch, ABC):
    def __init__(
            self,
            acquisition_function: AbstractAcquisitionFunction,
            config_space: ConfigurationSpace,
            rng: Union[bool, np.random.RandomState] = None,
            max_steps: Optional[int] = None,
            n_steps_plateau_walk: int = 10,
            n_sls_iterations: int = 10

    ):
        super().__init__(acquisition_function, config_space, rng)
        self.random_search = RandomSearch(
            acquisition_function=acquisition_function,
            config_space=config_space,
            rng=rng
        )
        self.local_search = FasterLocalSearch(
            acquisition_function=acquisition_function,
            config_space=config_space,
            rng=rng,
            max_steps=max_steps,
            n_steps_plateau_walk=n_steps_plateau_walk
        )
        self.n_sls_iterations = n_sls_iterations
