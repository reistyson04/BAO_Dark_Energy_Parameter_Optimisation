"""
BAO Dark-Energy Parameter Optimisation
=======================================

Demonstration of cosmological parameter inference using simplified
mock baryon acoustic oscillation (BAO) angular-residual measurements.

The script fits the two-parameter dark-energy model

    w(a) = w_p + w_a (1 - a)

by minimising chi-squared relative to a Lambda-CDM reference cosmology.

Workflow
--------
1. Import simplified mock BAO measurements.
2. Calculate cosmological distances for a trial (w_p, w_a).
3. Convert distances to predicted BAO angular scales.
4. Compare the predicted angular residuals with the mock measurements.
5. Minimise chi-squared to obtain the best-fit cosmology.
6. Construct the two-dimensional Delta-chi-squared surface.
7. Estimate joint 1-sigma and 2-sigma confidence regions.
8. Save the numerical results and confidence-contour figure.

The mock data are illustrative only. They are not Euclid observations
and should not be interpreted as realistic Euclid forecast constraints
or measurements of the Universe.
"""


# ============================================================
# IMPORTS
# ============================================================

import time

import matplotlib.pyplot as plt
import numpy as np

from scipy.integrate import cumulative_trapezoid
from scipy.optimize import differential_evolution
from scipy.stats import chi2 as chi2_distribution

from Mock_Euclid_BAO_Data import (
    Delta as angular_uncertainty,
    euclid_data as mock_euclid_data,
)


# ============================================================
# COSMOLOGICAL CONSTANTS
# ============================================================

c = 2.998e5       # Speed of light [km/s]
H0 = 67.37        # Hubble constant [km/s/Mpc]

Omega_m = 0.315
Omega_x = 0.685

BAO_SCALE = 150   # BAO standard-ruler scale [Mpc]


# ============================================================
# ANALYSIS CONFIGURATION
# ============================================================

# Number of points in the common numerical integration grid.
N_INTEGRATION = 3000

# Resolution of the final chi-squared surface.
N_WP = 151
N_WA = 151

# Search bounds for the dark-energy parameters.
WP_BOUNDS = (-1.0, -0.9)
WA_BOUNDS = (-0.35, 0.35)

BOUNDS = [
    WP_BOUNDS,
    WA_BOUNDS,
]

N_PARAMETERS = 2


# ============================================================
# DATA PREPARATION
# ============================================================

mock_data = np.asarray(
    mock_euclid_data,
    dtype=float,
)

z_data = mock_data[:, 0]

# Mock observed angular BAO residuals relative to Lambda-CDM.
theta_obs = mock_data[:, 1]

# Convert redshift to scale factor.
a_data = 1.0 / (1.0 + z_data)

# Allow the imported uncertainty to be either a scalar or array.
sigma_theta = np.asarray(
    angular_uncertainty,
    dtype=float,
)


# ============================================================
# INTEGRATION GRID
# ============================================================

# A common scale-factor grid is used for every model evaluation.
# This avoids performing a separate Python integration loop for
# every redshift and parameter combination.

a_grid = np.linspace(
    a_data.min(),
    1.0,
    N_INTEGRATION,
)


# ============================================================
# COSMOLOGICAL DISTANCE
# ============================================================

def distance(w_p, w_a):
    """
    Calculate the cosmological distance for all mock-data redshifts.

    The integrand is evaluated once on a common scale-factor grid
    and cumulatively integrated. The cumulative solution is then
    interpolated at each observed scale factor.

    Parameters
    ----------
    w_p : float
        Present-value dark-energy equation-of-state parameter.

    w_a : float
        Evolution parameter in

            w(a) = w_p + w_a (1 - a)

    Returns
    -------
    numpy.ndarray
        Distance evaluated at each mock-data redshift.
    """

    dark_energy = (
        Omega_x
        * a_grid ** (-3.0 * (w_p + w_a))
        * np.exp(
            3.0 * w_a * (a_grid - 1.0)
        )
    )

    integrand = (
        (c / H0)
        / np.sqrt(a_grid)
        / np.sqrt(
            Omega_m + dark_energy
        )
    )

    # Integral from the minimum scale factor to every point
    # on the common integration grid.
    cumulative = cumulative_trapezoid(
        integrand,
        a_grid,
        initial=0.0,
    )

    total = cumulative[-1]

    # Interpolate the cumulative integral at every mock-data
    # scale factor.
    cumulative_at_a = np.interp(
        a_data,
        a_grid,
        cumulative,
    )

    # Required integral is from a -> 1.
    return total - cumulative_at_a


# ============================================================
# BAO MODEL
# ============================================================

def bao_angle(distance_values):
    """
    Convert cosmological distance into the BAO angular scale.

    Parameters
    ----------
    distance_values : numpy.ndarray
        Cosmological distance values.

    Returns
    -------
    numpy.ndarray
        Predicted BAO angular scale.
    """

    return BAO_SCALE / distance_values


# Lambda-CDM reference cosmology.
#
# This only needs to be evaluated once because the reference
# cosmology remains fixed throughout the parameter search.

distance_lambda = distance(
    -1.0,
    0.0,
)

theta_lambda = bao_angle(
    distance_lambda
)


def model_delta_theta(w_p, w_a):
    """
    Calculate the BAO angular residual relative to Lambda-CDM.

    Parameters
    ----------
    w_p : float
        Present-value dark-energy parameter.

    w_a : float
        Dark-energy evolution parameter.

    Returns
    -------
    numpy.ndarray
        Difference between the trial-model BAO angular scale and
        the Lambda-CDM reference angular scale.
    """

    theta_model = bao_angle(
        distance(
            w_p,
            w_a,
        )
    )

    return theta_model - theta_lambda


# ============================================================
# CHI-SQUARED
# ============================================================

def chi_squared(parameters):
    """
    Calculate the total chi-squared for a trial cosmology.

    Parameters
    ----------
    parameters : sequence of float
        Pair containing (w_p, w_a).

    Returns
    -------
    float
        Total chi-squared value.
    """

    w_p, w_a = parameters

    delta_theta_model = model_delta_theta(
        w_p,
        w_a,
    )

    residuals = (
        theta_obs - delta_theta_model
    ) / sigma_theta

    return np.sum(
        residuals**2
    )


# ============================================================
# MAIN ANALYSIS
# ============================================================

def main():

    start_time = time.perf_counter()

    # ========================================================
    # FIND BEST-FIT PARAMETERS
    # ========================================================

    result = differential_evolution(
        chi_squared,
        bounds=BOUNDS,
        tol=1e-9,
        polish=True,
        seed=42,
    )

    w_p_best, w_a_best = result.x

    chi2_min = result.fun


    # ========================================================
    # GOODNESS OF FIT
    # ========================================================

    degrees_of_freedom = (
        len(mock_data)
        - N_PARAMETERS
    )

    reduced_chi2 = (
        chi2_min
        / degrees_of_freedom
    )

    print(
        "\nBest-fit cosmology"
    )

    print(
        "------------------"
    )

    print(
        f"w_p = {w_p_best:.6f}"
    )

    print(
        f"w_a = {w_a_best:.6f}"
    )

    print(
        f"Minimum χ² = {chi2_min:.6f}"
    )

    print(
        f"Degrees of freedom = "
        f"{degrees_of_freedom}"
    )

    print(
        f"Reduced χ² = "
        f"{reduced_chi2:.6f}"
    )


    # ========================================================
    # CONFIDENCE LEVELS
    # ========================================================

    # Joint confidence regions for two fitted parameters.

    delta_chi2_1sigma = (
        chi2_distribution.ppf(
            0.6827,
            df=2,
        )
    )

    delta_chi2_2sigma = (
        chi2_distribution.ppf(
            0.9545,
            df=2,
        )
    )

    print(
        "\nConfidence thresholds"
    )

    print(
        "---------------------"
    )

    print(
        f"1σ: Δχ² = "
        f"{delta_chi2_1sigma:.4f}"
    )

    print(
        f"2σ: Δχ² = "
        f"{delta_chi2_2sigma:.4f}"
    )


    # ========================================================
    # CHI-SQUARED SURFACE
    # ========================================================

    w_p_values = np.linspace(
        WP_BOUNDS[0],
        WP_BOUNDS[1],
        N_WP,
    )

    w_a_values = np.linspace(
        WA_BOUNDS[0],
        WA_BOUNDS[1],
        N_WA,
    )

    chi2_surface = np.empty(
        (
            N_WA,
            N_WP,
        )
    )

    for i, w_a in enumerate(
        w_a_values
    ):

        for j, w_p in enumerate(
            w_p_values
        ):

            chi2_surface[
                i,
                j,
            ] = chi_squared(
                (
                    w_p,
                    w_a,
                )
            )

    delta_chi2_surface = (
        chi2_surface
        - chi2_min
    )


    # ========================================================
    # CONFIDENCE REGIONS
    # ========================================================

    W_P, W_A = np.meshgrid(
        w_p_values,
        w_a_values,
    )

    one_sigma_mask = (
        delta_chi2_surface
        <= delta_chi2_1sigma
    )

    two_sigma_mask = (
        delta_chi2_surface
        <= delta_chi2_2sigma
    )

    w_p_one_sigma = W_P[
        one_sigma_mask
    ]

    w_a_one_sigma = W_A[
        one_sigma_mask
    ]

    w_p_two_sigma = W_P[
        two_sigma_mask
    ]

    w_a_two_sigma = W_A[
        two_sigma_mask
    ]


    # ========================================================
    # PROJECTED PARAMETER RANGES
    # ========================================================

    print(
        "\nProjected parameter ranges"
    )

    print(
        "--------------------------"
    )

    print(
        "w_p 1σ: "
        f"{w_p_one_sigma.min():.5f} "
        f"to "
        f"{w_p_one_sigma.max():.5f}"
    )

    print(
        "w_p 2σ: "
        f"{w_p_two_sigma.min():.5f} "
        f"to "
        f"{w_p_two_sigma.max():.5f}"
    )

    print(
        "w_a 1σ: "
        f"{w_a_one_sigma.min():.5f} "
        f"to "
        f"{w_a_one_sigma.max():.5f}"
    )

    print(
        "w_a 2σ: "
        f"{w_a_two_sigma.min():.5f} "
        f"to "
        f"{w_a_two_sigma.max():.5f}"
    )


    # ========================================================
    # BEST-FIT MODEL OUTPUT
    # ========================================================

    distance_best = distance(
        w_p_best,
        w_a_best,
    )

    theta_best = bao_angle(
        distance_best
    )

    delta_theta_best = (
        theta_best
        - theta_lambda
    )

    residuals_best = (
        theta_obs
        - delta_theta_best
    )

    chi2_contributions = (
        residuals_best
        / sigma_theta
    ) ** 2


    # ========================================================
    # PLOT CONFIDENCE CONTOURS
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    confidence_contours = ax.contour(
        W_P,
        W_A,
        delta_chi2_surface,
        levels=[
            delta_chi2_1sigma,
            delta_chi2_2sigma,
        ],
        linestyles=[
            "-",
            "--",
        ],
    )

    ax.clabel(
        confidence_contours,
        inline=True,
        fmt={
            delta_chi2_1sigma: "1σ",
            delta_chi2_2sigma: "2σ",
        },
    )

    # Best-fit point.
    ax.scatter(
        w_p_best,
        w_a_best,
        marker="o",
        label="Best fit",
        zorder=5,
    )

    # Lambda-CDM reference point.
    ax.scatter(
        -1.0,
        0.0,
        marker="x",
        label="ΛCDM",
        zorder=5,
    )

    ax.set_xlabel(
        r"$w_p$",
        fontsize=14,
    )

    ax.set_ylabel(
        r"$w_a$",
        fontsize=14,
    )

    ax.set_title(
        "BAO dark-energy constraints"
    )

    ax.tick_params(
        axis="both",
        labelsize=11,
    )

    ax.grid(
        alpha=0.15
    )

    ax.legend()

    plt.tight_layout()

    plt.savefig(
        "bao_confidence_contours.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()


    # ========================================================
    # SAVE NUMERICAL RESULTS
    # ========================================================

    np.savez(
        "chi2_optimisation_results.npz",

        w_p_best=w_p_best,
        w_a_best=w_a_best,

        chi2_min=chi2_min,
        reduced_chi2=reduced_chi2,

        degrees_of_freedom=(
            degrees_of_freedom
        ),

        w_p_values=w_p_values,
        w_a_values=w_a_values,

        chi2_surface=chi2_surface,

        delta_chi2_surface=(
            delta_chi2_surface
        ),

        delta_chi2_1sigma=(
            delta_chi2_1sigma
        ),

        delta_chi2_2sigma=(
            delta_chi2_2sigma
        ),

        distance_lambda=(
            distance_lambda
        ),

        theta_lambda=theta_lambda,

        distance_best=distance_best,

        theta_best=theta_best,

        delta_theta_best=(
            delta_theta_best
        ),

        residuals_best=(
            residuals_best
        ),

        chi2_contributions=(
            chi2_contributions
        ),
    )


    # ========================================================
    # RUNTIME
    # ========================================================

    elapsed_time = (
        time.perf_counter()
        - start_time
    )

    print(
        f"\nCompleted in "
        f"{elapsed_time:.2f} seconds."
    )

    print(
        "\nSaved:"
    )

    print(
        "  chi2_optimisation_results.npz"
    )

    print(
        "  bao_confidence_contours.png"
    )


# ============================================================
# RUN SCRIPT
# ============================================================

if __name__ == "__main__":
    main()