"""
Mock Euclid BAO Data
====================

Simplified mock baryon acoustic oscillation (BAO) measurements used
to demonstrate the dark-energy parameter inference procedure in
BAO_DE_Parameter_Optimisation.py.

Each row contains:

    [redshift, angular BAO residual]

where the angular residual represents the mock difference in the BAO
angular scale relative to the reference Lambda-CDM cosmology.

These values are illustrative mock data only. They are not Euclid
observations and are not intended to reproduce realistic Euclid
forecast constraints or measurements of the Universe.

The common uncertainty assigned to each mock measurement is given by
Delta.
"""


# ============================================================
# MOCK MEASUREMENT UNCERTAINTY
# ============================================================

# Common uncertainty assigned to each mock angular residual.

Delta = 0.00005


# ============================================================
# MOCK BAO DATA
# ============================================================

# Columns:
#
#   0 : redshift, z
#   1 : mock angular BAO residual, Delta theta
#
# Delta theta is measured relative to the Lambda-CDM reference
# cosmology used by the parameter-optimisation analysis.

euclid_data = [
    [0.3,  0.000342],
    [0.4,  0.000249],
    [0.5,  0.000228],
    [0.6,  0.000141],
    [0.7,  0.000127],
    [0.8,  0.000062],
    [0.9,  0.000118],
    [1.0,  0.000099],
    [1.1,  0.000028],
    [1.2,  0.000020],
    [1.3,  0.000015],
    [1.4,  0.000047],
    [1.5,  0.000049],
    [1.6, -0.000018],
    [1.7,  0.000023],
    [1.8,  0.000001],
    [1.9, -0.000040],
    [2.0, -0.000177],
]