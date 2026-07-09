# BBO Capstone Project — Bayesian Optimization of 8 Black-Box Functions

**Author:** Geoffrey [add surname]
**Programme:** Postgraduate ML Programme — Black-Box Optimization Capstone
**Status:** Complete (22 query rounds, Modules 12–24)

---

## 📌 Project Summary (non-technical)

This project tackled eight "mystery" mathematical functions with unknown
formulas, each accepting several numbers as input and returning one score.
The only way to learn about them was to guess an input and see the score —
and each guess was expensive, allowed just once per function per week over
22 rounds. Using **Bayesian Optimization**, the same technique used to tune
expensive machine learning models and design lab experiments, the project
built a statistical model that learned from every guess and decided where to
guess next, balancing exploring the unknown against refining what already
worked. The standout result: one function's score rose over 30-fold (258 to
8,662) after the model discovered a hidden high-performing region. Combined
improvement across all functions: **+8,418 points**.

---

## 🎯 Project Objectives

- Optimise 8 unknown black-box functions (`f: [0,1]^d → R`, d = 2–8) under a
  strict budget of **one query per function per round**.
- Ground every methodological choice in established Bayesian Optimization
  literature (see [`REFERENCES.md`](REFERENCES.md)).
- Document the process transparently enough that another researcher could
  reproduce the queries from the recorded data and code.

## 🧠 Method Overview

**Core approach:** Gaussian Process (GP) surrogate models with Expected
Improvement (EI) / Upper Confidence Bound (UCB) acquisition functions
(Jones et al., 1998; Snoek et al., 2012; Srinivas et al., 2010).

The strategy evolved considerably across the 22 rounds — from a single
uniform configuration applied to all 8 functions, to a fully adaptive,
per-function system incorporating:
- Hyperparameter tuning via grid search + cross-validation (Module 18)
- Cluster analysis of high-performing regions (Module 22)
- Dimensionality/variance analysis in the spirit of PCA to identify which
  input dimensions actually drove outcomes (Module 23)

Full methodological narrative and code: see
[`notebooks/BBO_Capstone_Analysis.ipynb`](notebooks/BBO_Capstone_Analysis.ipynb).

## 📊 Results

| Function | Dimensions | Initial best | Final best | Total gain |
|---|---|---|---|---|
| 1 | 2 | 0.00 | 0.00 | +0.00 |
| 2 | 2 | 0.61 | 0.69 | +0.08 |
| 3 | 3 | -0.03 | -0.01 | +0.03 |
| 4 | 4 | -10.07 | 0.57 | +10.64 |
| 5 | 4 | 258.00 | 8,662.48 | **+8,404.48** |
| 6 | 5 | -0.71 | -0.17 | +0.54 |
| 7 | 6 | 1.36 | 2.75 | +1.39 |
| 8 | 8 | 8.61 | 9.99 | +1.38 |

**Total cumulative gain: +8,418.34**

Full result data: [`data/final_results_summary.json`](data/final_results_summary.json)

## 📁 Repository Structure

```
BBO-Capstone/
├── README.md                          <- you are here
├── DATASHEET.md                       <- dataset documentation
├── MODEL_CARD.md                      <- model documentation
├── REFERENCES.md                      <- academic bibliography
├── requirements.txt                   <- pinned dependencies
├── data/
│   └── final_results_summary.json     <- consolidated query results
├── src/
│   └── bbo_optimizer.py               <- core GP-BO implementation (reusable)
└── notebooks/
    └── BBO_Capstone_Analysis.ipynb    <- full narrative, code, and figures
```

## ⚠️ Data Availability

The true black-box functions belong to the course platform and are **not**
redistributable. This repository contains only:
1. The query points we submitted and the scores we received back
   (`data/final_results_summary.json`), and
2. The code used to generate those queries (`src/`, `notebooks/`).

No large or proprietary datasets are hosted in this repository. See
[`DATASHEET.md`](DATASHEET.md) for full provenance details.

## 🔁 Reproducibility

All random elements use a fixed seed (`random_state=42`). Dependencies are
pinned in `requirements.txt`. To reproduce the demonstration in the notebook:

```bash
pip install -r requirements.txt
jupyter notebook notebooks/BBO_Capstone_Analysis.ipynb
```

## 📚 Documentation

- [`DATASHEET.md`](DATASHEET.md) — dataset documentation (Gebru et al., 2018 template)
- [`MODEL_CARD.md`](MODEL_CARD.md) — model documentation (Mitchell et al., 2019 template)
- [`REFERENCES.md`](REFERENCES.md) — full academic bibliography

## 📝 License

MIT License (or update to match your institution's requirements).
