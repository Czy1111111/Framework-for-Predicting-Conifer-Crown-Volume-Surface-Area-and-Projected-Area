# Framework-for-Predicting-Conifer-Crown-Volume-Surface-Area-and-Projected-Area

This repository provides the trained model and inference scripts used in our study for predicting high-dimensional conifer crown phenotypes from conventional forest inventory measurements.

The proposed framework integrates latent structural representations learned from synthetic 3D crown models with physics-informed multi-task prediction to estimate crown volume (V), surface area (A), and projected area (Aproj) without requiring direct three-dimensional observations.

---

## Repository Contents

| File | Description |
|------|-------------|
| `best_joint_model.pth` | Trained PyTorch weights of the final joint prediction model. |
| `test.py` | Python script for validating the trained model using real tree data. |
| `tree_params_real.csv` | Real tree measurements used for model evaluation. |

---

## Requirements

- Python >= 3.8
- PyTorch >= 1.12
- NumPy
- Pandas
- tqdm (optional)

Install the required packages using:

```bash
pip install torch numpy pandas tqdm
```

---

## Installation

Clone this repository:

```bash
git clone https://github.com/Czy1111111/Framework-for-Predicting-Conifer-Crown-Volume-Surface-Area-and-Projected-Area.git

cd Framework-for-Predicting-Conifer-Crown-Volume-Surface-Area-and-Projected-Area
```

---

## Input Data

The inference script requires a CSV file containing conventional tree inventory measurements.

The provided example dataset is:

```
tree_params_real.csv
```

The input variables include the basic tree measurements used by the model, such as:

- Tree height
- Diameter at breast height (DBH)
- Crown widths
- Height to crown base
- Height of maximum crown width

Please ensure that the input data follow the same variable names and formats used in the provided example file.

---

## Usage

Run the validation script using:

```bash
python test.py
```

The script will automatically:

- Load the trained model (`best_joint_model.pth`);
- Read the real tree dataset (`tree_params_real.csv`);
- Predict crown volume (V), surface area (A), and projected area (Aproj);
- Evaluate prediction performance on the real dataset.

---

## Output

The script reports prediction performance metrics, including:

- Root Mean Square Error (RMSE)
- Mean Absolute Error (MAE)
- Coefficient of Determination (R²)
- Relative Bias (RelBias)

Predicted values and evaluation results can also be saved by modifying the output settings in `test.py`.

---

## Model Description

The released model is the final version of the proposed structure-aware framework. It combines:

- latent structural representations derived from PCP-MAE,
- multi-task joint prediction,
- physics-informed constraints,
- and structure-aware calibration strategies.

The framework enables the recovery of high-dimensional crown phenotypes from low-cost forest inventory measurements while preserving geometric consistency among crown traits.

---

## Notes

This repository provides the trained model and inference code used for evaluation on real trees.

Training scripts and synthetic data generation procedures are not included in the current release.

---



## License

This repository is released for academic and research purposes only.
