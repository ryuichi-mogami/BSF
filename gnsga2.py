import numpy as np
import warnings
import sys
import os
sys.path.insert(0, os.path.abspath("../pymoo"))
from pymoo.algorithms.base.genetic import GeneticAlgorithm
from pymoo.docs import parse_doc_string
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from BSF.rank_and_crowding_drs import RankAndCrowding_g_DRS
from pymoo.util.nds.non_dominated_sorting import find_non_dominated
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.selection.tournament import compare, TournamentSelection
from pymoo.termination.default import DefaultMultiObjectiveTermination
from pymoo.util.display.multi import MultiObjectiveOutput
from pymoo.util.dominator import Dominator
from pymoo.util.misc import has_feasible
from pymoo.operators.selection.rnd import RandomSelection
from pymoo.core.population import Population
# ---------------------------------------------------------------------------------------------------------
# Binary Tournament Selection Function
# ---------------------------------------------------------------------------------------------------------

def normalize(F, ideal, nadir, eps=1e-12, clip=True):
    F = np.asarray(F, dtype=float)
    ideal = np.asarray(ideal, dtype=float)
    nadir = np.asarray(nadir, dtype=float)

    denom = nadir - ideal

    # 1) denom が小さすぎる次元を eps に置換（ゼロ割り防止）
    denom_safe = np.where(np.abs(denom) < eps, eps, denom)
    X = (F - ideal) / denom_safe

    return X

def _is_one_sided(F, z):
    less = np.any(F < z)
    more = np.any(F > z)
    return (less ^ more)

def _g_transform(F, z, penalty=5.0):
    return F if _is_one_sided(F, z) else (F + penalty)

def binary_tournament(pop, P, algorithm, **kwargs):
    n_tournaments, n_parents = P.shape

    if n_parents != 2:
        raise ValueError("Only implemented for binary tournament!")

    tournament_type = algorithm.tournament_type
    S = np.full(n_tournaments, np.nan)

    for i in range(n_tournaments):

        a, b = P[i, 0], P[i, 1]
        a_cv, a_f, b_cv, b_f = pop[a].CV[0], pop[a].F, pop[b].CV[0], pop[b].F
        rank_a, cd_a = pop[a].get("rank", "crowding")
        rank_b, cd_b = pop[b].get("rank", "crowding")

        # if at least one solution is infeasible
        if a_cv > 0.0 or b_cv > 0.0:
            S[i] = compare(a, a_cv, b, b_cv, method='smaller_is_better', return_random_if_equal=True)

        # both solutions are feasible
        else:

            if tournament_type == 'comp_by_dom_and_crowding':
                rel = Dominator.get_relation(a_f, b_f)
                if rel == 1:
                    S[i] = a
                elif rel == -1:
                    S[i] = b

            elif tournament_type == 'comp_by_rank_and_crowding':
                S[i] = compare(a, rank_a, b, rank_b, method='smaller_is_better')

            else:
                raise Exception("Unknown tournament type.")

            # if rank or domination relation didn't make a decision compare by crowding
            if np.isnan(S[i]):
                S[i] = compare(a, cd_a, b, cd_b, method='larger_is_better', return_random_if_equal=True)

    return S[:, None].astype(int, copy=False)


# ---------------------------------------------------------------------------------------------------------
# Survival Selection
# ---------------------------------------------------------------------------------------------------------


class RankAndCrowdingSurvival(RankAndCrowding_g_DRS):
    
    def __init__(self, norm, alpha=0, nds=None, crowding_func="cd"):
        super().__init__(alpha, nds, crowding_func)
        self._ref_point_cache = {}
        self.norm = norm

    def _load_ref_point(self, n_obj: int, problem_name: str) -> np.ndarray:
        sflag = False
        if problem_name[0] == 'S':
            problem_name = problem_name[6:]  # 'SDTLZ1' -> 'DTLZ1'
            sflag = True
        key = (n_obj, problem_name) 
        if key in self._ref_point_cache:
            return self._ref_point_cache[key]
        ref_file = f"/home/mogami/roi/ref_point_data/roi-p/m{n_obj}_{problem_name}_type1.csv" 
        ref_point = np.loadtxt(ref_file, delimiter=",", dtype=float) 
        if sflag == True:
            for i in range(len(ref_point)):
                ref_point[i] = pow(10, i)*ref_point[i]
        self._ref_point_cache[key] = ref_point  
        return ref_point

    def _do(self, problem, pop, n_survive=None, **kwargs):

        # 元の目的値（raw）を必ず保持
        F = pop.get("F").astype(float, copy=False)
        _, n_obj = F.shape
        ref_point = self._load_ref_point(n_obj, problem.name())
        
        # if self.norm:
        #     F_nondom = find_non_dominated(F)
        #     F_nd = F[F_nondom]
        #     # 理想点・ナディア点の計算
        #     ideal = F_nd.min(axis=0)
        #     nadir = F_nd.max(axis=0)
        #     # 正規化
        #     F_norm = normalize(F, ideal, nadir)
        #     ref_point = normalize(ref_point, ideal, nadir)
        #     pop.set("F", F_norm)
        
        survivors = super()._do(problem, pop, ref_point, n_survive=n_survive, **kwargs)

        # 必ず元の目的値に戻す（他の処理が元のFを前提にするため）
        pop.set("F", F)
        return survivors

# =========================================================================================================
# Implementation
# =========================================================================================================


class gNSGA2(GeneticAlgorithm):

    def __init__(self,
                 pop_size=100,
                 sampling=FloatRandomSampling(),
                 selection=RandomSelection(),
                #  selection=TournamentSelection(func_comp=binary_tournament),
                 crossover=SBX(eta=15, prob=0.9),
                 mutation=PM(eta=20),
                #  survival=RankAndCrowding(),
                 survival=None,
                 output=MultiObjectiveOutput(),
                 n_obj=None,
                 problem_name=None,
                 norm=False,
                 g_penalty=5.0,
                 alpha = 0,
                 **kwargs):
        self.g_penalty = float(g_penalty)

        if survival is None:
            if norm:
                survival = RankAndCrowdingSurvival(norm, alpha = alpha, crowding_func="cd-norm")
                print("norm")
            else:   
                survival = RankAndCrowdingSurvival(norm, alpha = alpha,  crowding_func="cd-no")
                print("un-norm")
        super().__init__(
            pop_size=pop_size,
            sampling=sampling,
            selection=selection,
            crossover=crossover,
            mutation=mutation,
            survival=survival,
            output=output,
            advance_after_initial_infill=True,
            **kwargs)

        self.termination = DefaultMultiObjectiveTermination()
        self.tournament_type = 'comp_by_dom_and_crowding'

    def _set_optimum(self, **kwargs):
        if not has_feasible(self.pop):
            self.opt = self.pop[[np.argmin(self.pop.get("CV"))]]
        else:
            self.opt = self.pop[self.pop.get("rank") == 0]

parse_doc_string(gNSGA2.__init__)
