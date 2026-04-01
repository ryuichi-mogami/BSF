import os
import sys
import click
import json


from bsf import BSF
from rank_and_crowding_drs import RankAndCrowdingDRS
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.algorithms.moo.spea2 import SPEA2

from pymoo.algorithms.moo.sms import LeastHypervolumeContributionSurvival
from pymoo.operators.selection.rnd import RandomSelection
from pymoo.algorithms.moo.nsga3 import ReferenceDirectionSurvival
from spea2_survival_drs import SPEA2SurvivalDRS
from ibea import IBEA, FitnessAssignment

from pymoo.util.ref_dirs import get_reference_directions
from pymoo.termination import get_termination

from pymoo.problems import get_problem
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter

from pymoo.decomposition.tchebicheff import Tchebicheff
from pymoo.decomposition.pbi import PBI

import numpy as np
import argparse

def run(n_obj, problem_name, alg, run_id, roi_type, ref_point, roi_radius): 
    emo_max_fess = 50000
    termination = get_termination("n_eval", emo_max_fess)    

    problem = get_problem(problem_name, n_obj=n_obj)

    algorithms = []
    if alg == "BNSGA2":
        if len(ref_point) == 1:
            algorithm1 = NSGA2(pop_size=100, selection=RandomSelection(),survival=BSF(roi_type=roi_type, ref_point=ref_point, roi_radius=roi_radius, inner_survival=RankAndCrowdingDRS(alpha=0)))
            algorithms = [algorithm1]
        elif len(ref_point) == 2:
            emo_max_fess = 25000
            get_termination("n_eval", emo_max_fess)
            algorithm1 = NSGA2(pop_size=100, selection=RandomSelection(),survival=BSF(roi_type=roi_type, ref_point=ref_point[0], roi_radius=roi_radius[0], inner_survival=RankAndCrowdingDRS(alpha=0)))
            algorithm2 = NSGA2(pop_size=100, selection=RandomSelection(),survival=BSF(roi_type=roi_type, ref_point=ref_point[1], roi_radius=roi_radius[1], inner_survival=RankAndCrowdingDRS(alpha=0)))
            algorithms = [algorithm1, algorithm2]
    elif alg == "BMNSGA2":
        if len(ref_point) == 1:
            algorithm1 = NSGA2(pop_size=100, selection=RandomSelection(),survival=BSF(roi_type=roi_type, ref_point=ref_point, roi_radius=roi_radius, inner_survival=RankAndCrowdingDRS(alpha=0.1)))
            algorithms = [algorithm1]
        elif len(ref_point) == 2:
            emo_max_fess = 25000
            get_termination("n_eval", emo_max_fess)
            algorithm1 = NSGA2(pop_size=100, selection=RandomSelection(),survival=BSF(roi_type=roi_type, ref_point=ref_point[0], roi_radius=roi_radius[0], inner_survival=RankAndCrowdingDRS(alpha=0.1)))
            algorithm2 = NSGA2(pop_size=100, selection=RandomSelection(),survival=BSF(roi_type=roi_type, ref_point=ref_point[1], roi_radius=roi_radius[1], inner_survival=RankAndCrowdingDRS(alpha=0.1)))
            algorithms = [algorithm1, algorithm2]
    elif alg == "BSMSEMOA":
        if len(ref_point) == 1:
            algorithm1 = SMSEMOA(pop_size=100, n_offspring=1, selection=RandomSelection(),survival=BSF(roi_type=roi_type,  ref_point=ref_point, roi_radius=roi_radius, inner_survival=LeastHypervolumeContributionSurvival()))
            algorithms = [algorithm1]
        elif len(ref_point) == 2:
            emo_max_fess = 25000
            get_termination("n_eval", emo_max_fess)
            algorithm1 = SMSEMOA(pop_size=100, n_offspring=1, selection=RandomSelection(),survival=BSF(roi_type=roi_type,  ref_point=ref_point[0], roi_radius=roi_radius[0], inner_survival=LeastHypervolumeContributionSurvival()))
            algorithm2 = SMSEMOA(pop_size=100, n_offspring=1, selection=RandomSelection(),survival=BSF(roi_type=roi_type,  ref_point=ref_point[1], roi_radius=roi_radius[1], inner_survival=LeastHypervolumeContributionSurvival()))
            algorithms = [algorithm1, algorithm2]
    elif alg == "BIBEA":
        if len(ref_point) == 1:
            algorithm1 = IBEA(pop_size=100, selection=RandomSelection(),survival=BSF(roi_type=roi_type, ref_point=ref_point, roi_radius=roi_radius, inner_survival=FitnessAssignment()))
            algorithms = [algorithm1]
        elif len(ref_point) == 2:
            emo_max_fess = 25000
            get_termination("n_eval", emo_max_fess)
            algorithm1 = IBEA(pop_size=100, selection=RandomSelection(),survival=BSF(roi_type=roi_type, ref_point=ref_point[0], roi_radius=roi_radius[0], inner_survival=FitnessAssignment()))
            algorithm2 = IBEA(pop_size=100, selection=RandomSelection(),survival=BSF(roi_type=roi_type, ref_point=ref_point[1], roi_radius=roi_radius[1], inner_survival=FitnessAssignment()))
            algorithms = [algorithm1, algorithm2]
    elif alg == "BNSGA3":
        if len(ref_point) == 1:
            ref_dirs = get_reference_directions("energy", n_obj, 100, seed=1)
            algorithm1 = NSGA3(ref_dirs, pop_size=100, selection=RandomSelection(),survival=BSF(roi_type=roi_type, ref_point=ref_point, roi_radius=roi_radius, inner_survival=ReferenceDirectionSurvival(ref_dirs)))
            algorithms = [algorithm1]
        elif len(ref_point) == 2:
            emo_max_fess = 25000
            get_termination("n_eval", emo_max_fess)
            ref_dirs = get_reference_directions("energy", n_obj, 100, seed=1)
            algorithm1 = NSGA3(ref_dirs, pop_size=100, selection=RandomSelection(),survival=BSF(roi_type=roi_type, ref_point=ref_point[0], roi_radius=roi_radius[0], inner_survival=ReferenceDirectionSurvival(ref_dirs)))
            algorithm2 = NSGA3(ref_dirs, pop_size=100, selection=RandomSelection(),survival=BSF(roi_type=roi_type, ref_point=ref_point[1], roi_radius=roi_radius[1], inner_survival=ReferenceDirectionSurvival(ref_dirs)))
            algorithms = [algorithm1, algorithm2]
    elif alg == "BSPEA2":
        if len(ref_point) == 1:
            algorithm1 = SPEA2(pop_size=100, selection=RandomSelection(),survival=BSF(roi_type=roi_type, ref_point=ref_point, roi_radius=roi_radius, inner_survival=SPEA2SurvivalDRS(alpha=0)))
            algorithms = [algorithm1]
        elif len(ref_point) == 2:
            emo_max_fess = 25000
            get_termination("n_eval", emo_max_fess)
            algorithm1 = SPEA2(pop_size=100, selection=RandomSelection(),survival=BSF(roi_type=roi_type, ref_point=ref_point[0], roi_radius=roi_radius[0], inner_survival=SPEA2SurvivalDRS(alpha=0)))
            algorithm2 = SPEA2(pop_size=100, selection=RandomSelection(),survival=BSF(roi_type=roi_type, ref_point=ref_point[1], roi_radius=roi_radius[1], inner_survival=SPEA2SurvivalDRS(alpha=0)))
            algorithms = [algorithm1, algorithm2]
    elif alg == "BMSPEA2":
        if len(ref_point) == 1:
            algorithm1 = SPEA2(pop_size=100, selection=RandomSelection(),survival=BSF(roi_type=roi_type, ref_point=ref_point, roi_radius=roi_radius, inner_survival=SPEA2SurvivalDRS(alpha=0.1)))
            algorithms = [algorithm1]
        elif len(ref_point) == 2:
            emo_max_fess = 25000
            get_termination("n_eval", emo_max_fess)
            algorithm1 = SPEA2(pop_size=100, selection=RandomSelection(),survival=BSF(roi_type=roi_type, ref_point=ref_point[0], roi_radius=roi_radius[0], inner_survival=SPEA2SurvivalDRS(alpha=0.1)))
            algorithm2 = SPEA2(pop_size=100, selection=RandomSelection(),survival=BSF(roi_type=roi_type, ref_point=ref_point[1], roi_radius=roi_radius[1], inner_survival=SPEA2SurvivalDRS(alpha=0.1)))
            algorithms = [algorithm1, algorithm2]
    else:
        print(f"{alg} is not available.")
        exit()

    print("Start minimize")
    for i, algorithm in enumerate(algorithms):
        res = minimize(problem, algorithm, termination, seed=run_id, verbose=False, save_history=True)
        idx = 50000 // 100 - 1
        F_gen = res.history[idx].pop.get("F") 
        res_dir_path = os.path.join(f'./results_ref{len(ref_point)}', f'{roi_type}/{alg}/{problem_name.upper()}/m{n_obj}') 
        os.makedirs(res_dir_path, exist_ok=True) 
        res_file_path = os.path.join(res_dir_path, f'pop_{run_id}th_run_{emo_max_fess}fevals_ref{i + 1}.csv') 
        np.savetxt(res_file_path, F_gen, delimiter=',') 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_obj', type=int)
    parser.add_argument('--problem_name', type=str)
    parser.add_argument('--alg', type=str)
    parser.add_argument('--roi_type', type=str)
    parser.add_argument('--ref_points', type=str)
    parser.add_argument('--roi_radius', type=str)
    parser.add_argument('--run_id', type=int)
    args = parser.parse_args()
    
    n_obj = args.n_obj   
    problem_name = args.problem_name
    alg = args.alg
    roi_type = args.roi_type
    ref_points = json.loads(args.ref_points)
    roi_radius = json.loads(args.roi_radius)
    run_id = args.run_id
    run(n_obj, problem_name, alg, run_id, roi_type, ref_points, roi_radius)