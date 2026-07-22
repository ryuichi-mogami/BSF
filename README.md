# BSF: Bi-criteria Selection Framework

BSF is a framework for incorporating preference information into evolutionary multi-objective optimization algorithms (EMOAs). Built on top of `pymoo`, it provides a unified framework for implementing and studying preference-based evolutionary multi-objective optimization (PBEMO) algorithms.

---

## Overview

BSF supports preference-based evolutionary multi-objective optimization using Regions of Interest (ROIs) defined with respect to reference points.

### Features

- Plug-in framework for incorporating preference information into EMOAs
- Support for multiple ROI definitions (e.g., ROI-C and ROI-P)
- Multiple reference points
- Visualization utilities for ROIs and optimization results
- Flexible specification of the objective space in which the ROI is defined

---

## Versioning

The latest documentation describes the current stable release.

For previous releases and version-specific changes, please refer to the **GitHub Releases** page.

---

## Requirements

- Python 3
- `pymoo==0.6.1.5`
- BSF repository

BSF is designed to work with a locally placed clone of `pymoo==0.6.1.5`.

Therefore, **`pymoo` and BSF must be cloned under the same parent directory.**

---

## Installation

Clone `pymoo` and BSF into the same parent directory.

```bash
git clone -b 0.6.1.5 https://github.com/anyoptimization/pymoo.git
git clone https://github.com/ryuichi-mogami/BSF.git
```

The directory structure should look like

```
parent_directory/
├── pymoo/
└── BSF/
```

---

## Usage

Move to the BSF directory and execute `test_pymoo.py`.

```bash
cd BSF

python3 test_pymoo.py \
  --n_obj 2 \
  --problem_name DTLZ1 \
  --alg BNSGA2 \
  --roi_type roi-c \
  --space "original_space" \
  --ref_points '[[0.2, 0.5], [0.8, 0.1]]' \
  --roi_radius '[[0.1, 0.1], [0.1, 0.1]]' \
  --run_id 0
```

---

## Visualization

BSF provides a visualization utility for inspecting the obtained solutions, reference points, and corresponding Regions of Interest.

For example,

```bash
python3 plot.py \
  --n_obj 2 \
  --problem_name DTLZ2 \
  --alg BNSGA2 \
  --roi_type roi-c \
  --space "original_space" \
  --ref_points '[[0.2, 0.5], [0.8, 0.1]]' \
  --roi_radius '[[0.1, 0.1], [0.1, 0.1]]' \
  --run_id 0
```

Example visualizations:

<p align="center">
  <img
    width="300"
    alt="ROI-C"
    src="https://github.com/user-attachments/assets/088d82b9-01e7-4c81-9f31-06d9bda5f2b9"
  />
  <img
    width="300"
    alt="ROI-P"
    src="https://github.com/user-attachments/assets/f40b00a6-1882-41e1-841c-f127b5f6c6f2"
  />
  <br>
  Example results on the DTLZ2 problem using BNSGA2.
  Left: ROI-C. Right: ROI-P.
</p>

---

## Command-Line Arguments

| Argument | Description |
|----------|-------------|
| `--n_obj` | Number of objectives. |
| `--problem_name` | Test problem name. |
| `--alg` | Optimization algorithm. |
| `--roi_type` | ROI definition (e.g., `roi-c`, `roi-p`). |
| `--space` | Presence or absence of normalization. |
| `--ref_points` | Reference point(s). |
| `--roi_radius` | Semi-axis lengths of each ROI. |
| `--run_id` | Random seed (run identifier). |

Example:

```text
--ref_points '[[0.2, 0.5], [0.8, 0.1]]'
```

defines two reference points.

```text
--roi_radius '[[0.1, 0.1], [0.1, 0.1]]'
```

defines the corresponding ROI radii.

---

---

## License

Please add an appropriate license (e.g., MIT License) before public distribution.
