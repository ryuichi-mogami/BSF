import os
import numpy as np

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
    # calculate the ASF value
    asf_values = np.full(len(F), np.inf)
    for i in range(len(F)):
        asf_values[i] = np.max(w[i]*(F[i] - ref_point))
    
    return asf_values

class BSF(Survival):
    """
    Parameters
    ----------
    roi_type : {"roi-c", "roi-a", "roi-p"}
    space : {"normalized_space", "original_space"}
    ref_point : array-like
        ref_point is originally defined in the original space, but if space is "normalized_space" it will be normalized internally.
    roi_radius : float
        if space is "normalized_space", roi_radius shoud be [0, 1], and space is "original_space", roi_radius should be in the original space.
    inner_survival : Survival
        Downstream survival operator, e.g. RankAndCrowding().
    
    """
    def __init__(self, 
                roi_type,
                space, 
                ref_point, 
                roi_radius,
                inner_survival):
        super().__init__()

        self.roi_type = roi_type
        self.ref_point = ref_point
        self.roi_radius = roi_radius
        self.space = space
        self.inner_survival = inner_survival

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

            asf_arr = np.full(len(F), np.inf)
            w = np.ones_like(F)
            asf_arr = asf(F, ref_point, w)

            pivot_idx = np.argmin(asf_arr)
            pivot_point = F[pivot_idx]
            dist_arr_pivot = F - pivot_point

            val = np.sum((dist_arr_pivot/roi_radius)**2, axis=1)
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

        # normalize the objective values if needed
        if self.space == "normalized_space":
            nd_idx = find_non_dominated(F)
            F_nd = F[nd_idx]
            ideal = np.min(F_nd, axis=0)
            nadir = np.max(F_nd, axis=0)
            F_work = normalize(F, ideal, nadir)
            ref_point_work = normalize(self.ref_point, ideal, nadir)
        elif self.space == "original_space":
            F_work = F
            ref_point_work = self.ref_point
        else:
            raise ValueError("Unknown space! {} is not supported.".format(self.space))
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

            return Population.create(*survivors)
        
        # if the number of solutions in the ROI is greater than n_survive
        trimmed_pop = pop[sel_mask]

        return self.inner_survival.do(problem=problem, pop=trimmed_pop, n_survive=n_survive)