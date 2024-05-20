"""Module of constants for Elasticity."""

LINEAR = "linear"
POWER = "power"
EXPONENTIAL = "exponential"

MODEL_TYPES = [LINEAR, POWER, EXPONENTIAL]

CV_SUFFIXES_CS = [
    "mean_a",
    "mean_b",
    "mean_elasticity",
    "mean_r2",
    "mean_relative_absolute_error",
    "mean_norm_rmse",
]

MODEL_SUFFIXES_CS = [
    "a",
    "b",
    "pvalue",
    "elasticity",
    "elasticity_error_propagation",
    "r2",
    "relative_absolute_error",
    "norm_rmse",
    "aic",
]

# Combine CV_SUFFIXES_CS and MODEL_SUFFIXES_CS to create MODEL_TYPE_SUFFIXE_CS
MODEL_TYPE_SUFFIXE_CS = CV_SUFFIXES_CS + MODEL_SUFFIXES_CS

# Generate OUTPUT_CS dynamically using list comprehensions
OUTPUT_CS = [
    f"{model_type}_{suffix}"
    for model_type in MODEL_TYPES
    for suffix in MODEL_TYPE_SUFFIXE_CS
]

# Add best model-specific keys using list comprehension
OUTPUT_CS += [f"best_model_{suffix}" for suffix in MODEL_SUFFIXES_CS]

# Add other specific keys directly
OUTPUT_CS.extend(
    [
        "median_quantity",
        "median_price",
        "quality_test",
        "quality_test_high",
        "quality_test_medium",
        "details",
        "uid",  # Including uid as in the original OUTPUT_CS
    ]
)
