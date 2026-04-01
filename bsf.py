import os
import numpy as np
import sys
sys.path.insert(0, os.path.abspath("../pymoo"))
from pymoo.core.population import Population
from pymoo.core.survival import Survival
from pymoo.util.nds.non_dominated_sorting import find_non_dominated

def normalize(F, ideal, nadir):
    F = np.asarray(F, dtype=float)
    ideal = np.asarray(ideal, dtype=float)
    nadir = np.asarray(nadir, dtype=float)

    denom = nadir - ideal
    denom = np.where(np.isclose(denom, 0.0), 1.0, denom)

    return (F - ideal) / denom

def asf(F, ref_point, w):
    F = np.asarray(F, dtype=float)
    ref_point = np.asarray(ref_point, dtype=float)
    w = np.asarray(w, dtype=float)

    return np.max(w * (F - ref_point), axis=1)

class BSF(Survival):
    """
    Parameters
    ----------
    roi_type : {"roi-c", "roi-a", "roi-p"}
    ref_point : array-like
        ref_point is originally defined in the original space.
    roi_radius : float
        roi_radius should be in the original space.
    inner_survival : Survival
        Downstream survival operator, e.g. RankAndCrowding().
    
    """
    def __init__(self, 
                roi_type,
                ref_point, 
                roi_radius,
                inner_survival):
        super().__init__()

        self.roi_type = roi_type
        self.ref_point = ref_point
        self.roi_radius = roi_radius
        self.inner_survival = inner_survival
        self.opt = []

    def select_roi(self, F, ref_point, roi_radius):
        if self.roi_type == "roi-c":
            nd_idx = find_non_dominated(F)

            # calculate the distance to the reference point for each solution
            dist_arr = np.full(len(F), np.inf)
            for i in nd_idx:
                dist_arr[i] = np.linalg.norm(F[i] - ref_point)
            
            # find the pivot solution
            pivot_idx = np.argmin(dist_arr)
            pivot_point = F[pivot_idx]

            # calculate the distance to the pivot point for each solution
            dist_arr_pivot = F - pivot_point

            # select solutions within the ROI
            val = np.sum((dist_arr_pivot/roi_radius)**2, axis=1)
            sel_mask = val <= 1.0
            
            # note: the dist_to_pivot is not used in the current implementation, but it can be used for tie-breaking in the future
            dist_to_pivot = np.linalg.norm(F - pivot_point, axis=1)
            return sel_mask, dist_to_pivot

        elif self.roi_type == "roi-a":
            nd_idx = find_non_dominated(F)

            w = np.ones(F.shape[1], dtype=float)

            asf_arr = asf(F[nd_idx], ref_point, w)

            pivot_local_idx = np.argmin(asf_arr)
            pivot_idx = nd_idx[pivot_local_idx]
            pivot_point = F[pivot_idx]

            dist_arr_pivot = F - pivot_point

            val = np.sum((dist_arr_pivot / roi_radius) ** 2, axis=1)
            sel_mask = val <= 1.0
            dist_to_pivot = np.linalg.norm(F - pivot_point, axis=1)
            return sel_mask, dist_to_pivot
        elif self.roi_type == "roi-p":
            less_eq = np.all(F <= ref_point, axis=1)
            greater_eq = np.all(F >= ref_point, axis=1)
            sel_mask = np.logical_or(less_eq, greater_eq)

            dist_to_ref = np.linalg.norm(F - ref_point, axis=1)
            return sel_mask, dist_to_ref
        else:
            raise ValueError("Unknown roi_type! {} is not supported.".format(self.roi_type))
        
    def _do(self, problem, pop, *args, n_survive=None, **kwargs):
        F = pop.get("F").astype(float, copy=False)
        F_work = F
        ref_point_work = self.ref_point
        # check whether F is in the ROI
        sel_mask, dist_metric = self.select_roi(F_work, ref_point_work, self.roi_radius)

        selected_idx = np.where(sel_mask)[0].tolist()
        R_in = len(selected_idx)

        # if the number of solutions in the ROI is less than or equal to n_survive
        if R_in <= n_survive:
            remaining = n_survive - R_in
            if remaining > 0:
                unselected_idx = np.where(~sel_mask)[0]
                order = unselected_idx[np.argsort(dist_metric[~sel_mask])]
                selected_idx.extend(order[:remaining].tolist())
            
            survivors = pop[selected_idx]
            survivors = Population.create(*survivors)
            self.opt = survivors
            return survivors
        
        # if the number of solutions in the ROI is greater than n_survive
        trimmed_pop = pop[sel_mask]
        survivors = self.inner_survival.do(problem=problem, pop=trimmed_pop, n_survive=n_survive)
        self.opt = survivors
        return survivors