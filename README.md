# BSF: Bi-criteria Selection Framework

BSF is a framework for incorporating preference information into evolutionary multi-objective optimization algorithms.
Built on top of `pymoo`, it is designed to provide a unified basis for implementing and studying PBEMO algorithms.

---

## Overview

BSF supports preference-based evolutionary multi-objective optimization using Regions of Interest (ROI) defined with respect to reference points.

---

## Requirements

* Python 3
* `pymoo==0.6.1.5`
* BSF repository

BSF is designed to work with a locally placed clone of `pymoo==0.6.1.5`.
Therefore, **`pymoo` and BSF must be cloned under the same parent directory.**

---

## Installation

Clone `pymoo` and BSF into the same parent directory as follows:

```bash
git clone -b 0.6.1.5 https://github.com/anyoptimization/pymoo.git
git clone https://github.com/ryuichi-mogami/BSF.git
```

---

## Usage

Move to the BSF directory and run `test_pymoo.py`:

```bash
cd BSF
python3 test_pymoo.py \
  --n_obj 2 \
  --problem_name "DTLZ1" \
  --alg "BNSGA2" \
  --roi_type "roi-c" \
  --ref_points '[[0.2, 0.5], [0.8, 0.1]]' \
  --roi_radius '[[0.1, 0.1], [0.1, 0.1]]' \
  --run_id 0
```

---

## Visualization

BSF also provides a visualization script for inspecting the relationship between the obtained solutions, reference points, and the corresponding regions of interest.

For example, the following command generates a plot for a two-objective DTLZ2 problem:

```bash
python3 plot.py \
  --n_obj 2 \
  --problem_name "DTLZ2" \
  --alg "BNSGA2" \
  --roi_type "roi-c" \
  --ref_points '[[0.2, 0.5], [0.8, 0.1]]' \
  --roi_radius '[[0.1, 0.1], [0.1, 0.1]]' \
  --run_id 0
```
An example visualization is shown below:
<p align="center">
  <img
    width="300"
    height="300"
    alt="DTLZ2_BNSGA2_roi-c_run0"
    src="https://github.com/user-attachments/assets/088d82b9-01e7-4c81-9f31-06d9bda5f2b9"
  />
  <img
    width="300"
    height="300"
    alt="DTLZ2_BNSGA2_roi-p_run0"
    src="https://github.com/user-attachments/assets/f40b00a6-1882-41e1-841c-f127b5f6c6f2"
  />
  <br>
  Example visualizations for the DTLZ2 problem using BNSGA2: ROI-C (left) and ROI-P (right).
</p>

---

## Meaning of the Example Arguments

* `--n_obj`  
  Sets the number of objectives.

* `--problem_name`  
  Specifies the test problem.

* `--alg`  
  Specifies the algorithm to be used.

* `--roi_type`  
  Specifies the ROI type.

* `--ref_points '[[0.2, 0.5], [0.8, 0.1]]'`  
  Specifies the reference points. 

* `--roi_radius '[[0.1, 0.1], [0.1, 0.1]]'`  
  Specifies the semi-axis lengths of the ROI associated with each reference point.

* `--run_id 0`

---

## Notes

This framework is intended for researchers working on preference-based evolutionary multi-objective optimization, particularly in settings where reference-point-based preference articulation is used to define one or more ROIs.
