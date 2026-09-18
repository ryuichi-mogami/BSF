import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath("../pymoo"))
from pymoo.operators.survival.rank_and_crowding import RankAndCrowding
from pymoo.util.randomized_argsort import randomized_argsort
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.core.survival import Survival, split_by_feasibility
from pymoo.core.population import Population
from pymoo.operators.survival.rank_and_crowding.metrics import get_crowding_function

class RankAndCrowdingDRS(RankAndCrowding):

    def __init__(self, alpha=0.1, nds=None, crowding_func="cd"):
        super().__init__(nds=nds, crowding_func=crowding_func)
        self.alpha = alpha

    def _do(self,
            problem,
            pop,
            *args,
            n_survive=None,
            **kwargs):

        F_raw = pop.get("F").astype(float, copy=False)

        f_mean = F_raw.mean(axis=1, keepdims=True)
        F = (1.0 - self.alpha) * F_raw + self.alpha * f_mean


        # the final indices of surviving individuals
        survivors = []

        # do the non-dominated sorting until splitting front
        fronts = self.nds.do(F, n_stop_if_ranked=n_survive)

        for k, front in enumerate(fronts):
            
            I = np.arange(len(front))

            # current front sorted by crowding distance if splitting
            if len(survivors) + len(I) > n_survive:

                # Define how many will be removed
                n_remove = len(survivors) + len(front) - n_survive

                # re-calculate the crowding distance of the front
                crowding_of_front = \
                    self.crowding_func.do(
                        F[front, :],
                        n_remove=n_remove
                    )

                I = randomized_argsort(crowding_of_front, order='descending', method='numpy')
                I = I[:-n_remove]

            # otherwise take the whole front unsorted
            else:
                # calculate the crowding distance of the front
                crowding_of_front = \
                    self.crowding_func.do(
                        F[front, :],
                        n_remove=0
                    )

            # save rank and crowding in the individual class
            for j, i in enumerate(front):
                pop[i].set("rank", k)
                pop[i].set("crowding", crowding_of_front[j])

            # extend the survivors by all or selected individuals
            survivors.extend(front[I])

        return pop[survivors]

class RankAndCrowding_g(Survival):

    def __init__(self,nds=None, crowding_func="cd"):
        
        crowding_func_ = get_crowding_function(crowding_func)

        super().__init__(filter_infeasible=True)
        self.nds = nds if nds is not None else NonDominatedSorting(method="fast_non_dominated_sort_g")
        self.crowding_func = crowding_func_

    def _do(self,
            problem,
            pop,
            z,
            *args,
            n_survive=None,
            **kwargs):

        # get the objective space values and objects
        F = pop.get("F").astype(float, copy=False)
        # the final indices of surviving individuals
        survivors = []

        # do the non-dominated sorting until splitting front
        fronts = self.nds.do(F, n_stop_if_ranked=n_survive,z=z)
        for k, front in enumerate(fronts):
            
            I = np.arange(len(front))

            # current front sorted by crowding distance if splitting
            if len(survivors) + len(I) > n_survive:

                # Define how many will be removed
                n_remove = len(survivors) + len(front) - n_survive

                # re-calculate the crowding distance of the front
                crowding_of_front = \
                    self.crowding_func.do(
                        F[front, :],
                        n_remove=n_remove
                    )

                I = randomized_argsort(crowding_of_front, order='descending', method='numpy')
                I = I[:-n_remove]

            # otherwise take the whole front unsorted
            else:
                # calculate the crowding distance of the front
                crowding_of_front = \
                    self.crowding_func.do(
                        F[front, :],
                        n_remove=0
                    )

            # save rank and crowding in the individual class
            for j, i in enumerate(front):
                pop[i].set("rank", k)
                pop[i].set("crowding", crowding_of_front[j])

            # extend the survivors by all or selected individuals
            survivors.extend(front[I])

        return pop[survivors]
        
class RankAndCrowding_g_DRS(Survival):

    def __init__(self, alpha = 0, nds=None, crowding_func="cd"):
        
        crowding_func_ = get_crowding_function(crowding_func)

        super().__init__(filter_infeasible=True)
        self.nds = nds if nds is not None else NonDominatedSorting(method="fast_non_dominated_sort_g")
        self.crowding_func = crowding_func_
        self.alpha = alpha

    def _do(self,
            problem,
            pop,
            z,
            *args,
            n_survive=None,
            **kwargs):

        # get the objective space values and objects
        F_raw = pop.get("F").astype(float, copy=False)

        f_mean = F_raw.mean(axis=1, keepdims=True)
        F = (1.0 - self.alpha) * F_raw + self.alpha * f_mean
        # the final indices of surviving individuals
        survivors = []

        # do the non-dominated sorting until splitting front
        fronts = self.nds.do(F, n_stop_if_ranked=n_survive,z=z)
        for k, front in enumerate(fronts):
            
            I = np.arange(len(front))

            # current front sorted by crowding distance if splitting
            if len(survivors) + len(I) > n_survive:

                # Define how many will be removed
                n_remove = len(survivors) + len(front) - n_survive

                # re-calculate the crowding distance of the front
                crowding_of_front = \
                    self.crowding_func.do(
                        F[front, :],
                        n_remove=n_remove
                    )

                I = randomized_argsort(crowding_of_front, order='descending', method='numpy')
                I = I[:-n_remove]

            # otherwise take the whole front unsorted
            else:
                # calculate the crowding distance of the front
                crowding_of_front = \
                    self.crowding_func.do(
                        F[front, :],
                        n_remove=0
                    )

            # save rank and crowding in the individual class
            for j, i in enumerate(front):
                pop[i].set("rank", k)
                pop[i].set("crowding", crowding_of_front[j])

            # extend the survivors by all or selected individuals
            survivors.extend(front[I])

        return pop[survivors]