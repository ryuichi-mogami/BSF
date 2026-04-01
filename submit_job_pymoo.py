#!/usr/bin/env python3
import subprocess
import os
import json

if __name__ == '__main__':
    count = 0

    base_dir = "/home/mogami/bsf"
    output_dir = f"{base_dir}/output/sh"
    os.makedirs(output_dir, exist_ok=True)

    # 確認条件を固定
    problem_name = "DTLZ1"
    n_obj = 2

    # test_pymoo.py 側が受けるアルゴリズム名に合わせる
    # 必要に応じて手元の実装名に置き換えること
    algorithms = [
        "BNSGA2",
        "BMNSGA2",
        "BIBEA",
        "BSMSEMOA",
        "BNSGA3",
        "BSPEA2",
        "BMSPEA2",
    ]

    # 単一参照点・複数参照点の両方を確認
    test_cases = [
        {
            "label": "ref1",
            "ref_points": [[0.8, 0.1]],
            "roi_radius": [[0.1, 0.1]],
        },
        {
            "label": "ref2",
            "ref_points": [[0.2, 0.5], [0.8, 0.1]],
            "roi_radius": [[0.1, 0.1], [0.1, 0.1]],
        },
    ]

    for roi_type in ["roi-c", "roi-p"]:
        for alg in algorithms:
            for case in test_cases:
                count += 1

                ref_points_str = json.dumps(case["ref_points"])
                roi_radius_str = json.dumps(case["roi_radius"])

                args = (
                    f'--n_obj {n_obj} '
                    f'--problem_name {problem_name} '
                    f'--alg {alg} '
                    f'--roi_type {roi_type} '
                    f"--ref_points '{ref_points_str}' "
                    f"--roi_radius '{roi_radius_str}' "
                    f'--run_id 0'
                )

                outfile = os.path.join(
                    output_dir,
                    f'{problem_name}_{alg}_{roi_type}_m{n_obj}_{case["label"]}.out'
                )

                print(f"submit: {args}")
                subprocess.run([
                    'qsub',
                    '-l', 'walltime=72:00:00',
                    '-j', 'oe',
                    '-o', outfile,
                    '-F', args,
                    'job_pymoo.sh'
                ], check=False)

    print(f"total submitted = {count}")