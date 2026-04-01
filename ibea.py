import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath("./pymoo"))
from pymoo.algorithms.base.genetic import GeneticAlgorithm
from pymoo.core.population import Population
from pymoo.core.survival import Survival
from pymoo.docs import parse_doc_string
# from pymoo.indicators.hv.exact import ExactHypervolume
# from pymoo.indicators.hv.exact_2d import ExactHypervolume2D
# from pymoo.indicators.hv.monte_carlo import ApproximateMonteCarloHypervolume
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.selection.tournament import compare, TournamentSelection
from pymoo.util.display.multi import MultiObjectiveOutput
from pymoo.util.dominator import Dominator
from pymoo.util.function_loader import load_function
# from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.normalization import normalize

from pymoo.operators.selection.rnd import RandomSelection

class FitnessAssignment(Survival):

    def __init__(self, kappa=0.05, bq_indicator="epsilon") -> None:
        super().__init__(filter_infeasible=True)
        self.kappa = kappa
        self.bq_indicator = bq_indicator

    def _do(self, problem, pop, *args, n_survive=None, ideal=None, nadir=None, **kwargs):

        # get the objective space values and objects
        F = pop.get("F").astype(float, copy=False)

        # if the boundary points are not provided -> estimate them from pop
        if ideal is None:
            ideal = F.min(axis=0)
        if nadir is None:
            nadir = F.max(axis=0)

        # the number of objectives
        _, n_obj = F.shape
        PQ_size = len(F)

        # the final indices of surviving individuals
        survivor_list = list(range(PQ_size))

        normalized_F = (F - ideal) / (nadir - ideal)

        if self.bq_indicator == "epsilon":
            # epsilon_matrix = np.zeros((pop_size, pop_size))    
            # for i, Fi in enumerate(normalized_F):
            #     for j, Fj in enumerate(normalized_F):
            #         # This should not be np.max(Fi - Fj)
            #         epsilon_value = np.max(Fj - Fi)
            #         epsilon_matrix[j][i] = epsilon_value            
            bqi_matrix = np.max(normalized_F[:, None, :] - normalized_F[None, :, :], axis=2)
        else:
            raise ValueError(f"{self.bq_indicator} is not available")
            
        bqi_max = np.max(bqi_matrix)
        fitness_arr = np.zeros(PQ_size)
        for i in range(PQ_size):
            for j in range(PQ_size):            
                if i != j:
                    fitness_arr[i] += -np.exp(-bqi_matrix[j][i] / (bqi_max * self.kappa))              
                    
        while len(survivor_list) > n_survive:
            # The worst individual is removed from P \cup Q
            worst_id = np.argmin(fitness_arr)
            fitness_arr[worst_id] = np.inf
            survivor_list.remove(worst_id) 
            # Update the fitness values
            for i in range(PQ_size):
                if i != worst_id:
                    fitness_arr[i] += np.exp(-bqi_matrix[worst_id][i] / (bqi_max * self.kappa))
                    
        for i in range(PQ_size):        
            pop[i].set("fitness", fitness_arr[i])
        
        survivors = pop[survivor_list]
        
        return Population.create(*survivors)


def binary_tournament(pop, P, algorithm, **kwargs):
    n_tournaments, n_parents = P.shape

    if n_parents != 2:
        raise ValueError("Only implemented for binary tournament!")

    S = np.full(n_tournaments, np.nan)

    for i in range(n_tournaments):
        a, b = P[i, 0], P[i, 1]
        a_cv, b_cv = pop[a].CV[0], pop[b].CV[0]
        fitness_a = pop[a].get("fitness")
        fitness_b = pop[b].get("fitness")
    
        # if at least one solution is infeasible
        if a_cv > 0.0 or b_cv > 0.0:
            S[i] = compare(a, a_cv, b, b_cv, method='smaller_is_better', return_random_if_equal=True)
        # both solutions are feasible
        else:
            if fitness_a > fitness_b:
                S[i] = a
            else:
                S[i] = b

    return S[:, None].astype(int, copy=False)

class IBEA(GeneticAlgorithm):

    def __init__(self,
                 pop_size=100,
                 sampling=FloatRandomSampling(),
                 #selection=binary_tournament(), #RandomSelection(),
                 selection=TournamentSelection(func_comp=binary_tournament),
                 crossover=SBX(),
                 mutation=PM(),
                 survival=FitnessAssignment(),
                 eliminate_duplicates=True,
                 n_offsprings=None,
                 normalize=True,
                 output=MultiObjectiveOutput(),
                 **kwargs):
        super().__init__(pop_size=pop_size,
                         sampling=sampling,
                         selection=selection,
                         crossover=crossover,
                         mutation=mutation,
                         survival=survival,
                         eliminate_duplicates=eliminate_duplicates,
                         n_offsprings=n_offsprings,
                         output=output,
                         advance_after_initial_infill=True,
                         **kwargs)

        self.normalize = normalize

    def _advance(self, infills=None, **kwargs):
        # merge the offsprings with the current population
        if infills is not None:
            pop = Population.merge(self.pop, infills)
        else:
            pop = self.pop
        
        self.pop = self.survival.do(self.problem, pop, n_survive=self.pop_size, algorithm=self,
                                    ideal=None, nadir=None, **kwargs)


parse_doc_string(IBEA.__init__)
