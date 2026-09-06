# BAO Dark-Energy Parameter Optimisation

A compact demonstration of cosmological parameter inference using simplified mock baryon acoustic oscillation (BAO) measurements.

The analysis fits the two-parameter dark-energy model

\[
w(a) = w_p + w_a(1-a)
\]

by comparing predicted BAO angular residuals against mock data and minimising chi-squared.

## Repository

- `Mock_Euclid_BAO_Data.py`  
  Simplified mock BAO angular-residual measurements and assumed uncertainty.

- `BAO_DE_Parameter_Optimisation.py`  
  Numerical distance calculation, chi-squared optimisation, confidence-region construction and plotting.

## Method

The analysis:

1. Computes cosmological distances for trial values of \(w_p\) and \(w_a\).
2. Converts the BAO standard ruler into a predicted angular scale.
3. Calculates the angular residual relative to a ΛCDM reference cosmology.
4. Minimises chi-squared to determine the best-fit parameters.
5. Constructs the two-dimensional \(\Delta\chi^2\) surface and joint 1σ and 2σ confidence regions.

## Data

The included BAO measurements are **simplified mock data used only to demonstrate the inference method**.

They are not Euclid observations and should not be interpreted as realistic Euclid forecast constraints or measurements of the Universe.

## Requirements

- NumPy
- SciPy
- Matplotlib

## Run

```bash
python BAO_DE_Parameter_Optimisation.py