# PriValue: Privacy-Preserving Data Valuation via DP Synthetic Data

This repository is the official implementation of **[PriValue: Privacy-Preserving Data Valuation via Differentially Private Dataset Synthesis](https://github.com/XN-IRIS/privalue)** (NeurIPS 2025 Submission).

<p align="center">
  <img src="overview.png" width="600"/>
</p>

> 🛡️ PriValue enables Shapley-based data valuation in collaborative ML without leaking sensitive data, using differentially private (DP) synthetic datasets and a fidelity–utility calibration mechanism.

---

## Requirements

To install dependencies:

```bash
conda create -n privalue python=3.9
conda activate privalue
pip install -r requirements.txt
```

> CIFAR-10 will be automatically downloaded during the first run.

---

## Dataset Partitioning

We support two synthesis frameworks:

* **PE**: 10 equal-sized partitions simulating 10 data providers.
* **PrivImage**: 5 data providers, each with 10,000 samples.

To prepare partitioned private data:

```bash
python synthesis/partition.py
```

The resulting files are saved in:

* `pe_pridata/usr0/` to `usr9/`
* `privimage_pridata/usr0/` to `usr4/`

---

## Synthetic Data Generation

We provide wrappers to integrate [PE](https://github.com/zinanlin/pe) and [PrivImage](https://github.com/Kecen/privimage) under the `/synthesis` folder. Refer to their respective READMEs for detailed setup.

---

## Valuation & Calibration

To run Shapley valuation and apply fidelity–utility calibration:

```bash
python valuation/run_shapley.py --method pe --eps 4.0
python valuation/apply_calibration.py --alpha 0.4 --beta 0.3 --gamma 0.3
```

---

## Results

We report the following improvements in valuation error and ranking correlation:

| Method    | ε   | Pre-Cal. Rel. Error | Post-Cal. Rel. Error | Spearman ↑ |
| --------- | --- | ------------------- | -------------------- | ---------- |
| PE        | 4.0 | 32.6%               | 6.7%                 | 0.95       |
| PrivImage | 4.0 | 38.7%               | 14.2%                | 1.00       |

---

## Pre-trained Synthetic Models

Due to privacy concerns, pre-generated synthetic datasets are not included in this repository. You may reproduce them following our instructions or use safe public surrogates.

---