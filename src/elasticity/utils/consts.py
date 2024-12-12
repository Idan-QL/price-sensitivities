"""Module of constants for Elasticity."""

CODE_VERSION = "2.0"

LINEAR = "linear"
POWER = "power"
EXPONENTIAL = "exponential"

MIN_ELASTICITY = -3.8
VERY_ELASTIC_THRESHOLD = -2.3
SUPER_ELASTIC_THRESHOLD = -3.8
ELASTIC_THRESHOLD = -1

# MODEL_TYPES = [POWER, EXPONENTIAL]
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

MODEL_TYPE_SUFFIXE_CS = CV_SUFFIXES_CS + MODEL_SUFFIXES_CS

# Generate OUTPUT_CS dynamically using list comprehensions
OUTPUT_CS = [
    f"{model_type}_{suffix}" for model_type in MODEL_TYPES for suffix in MODEL_TYPE_SUFFIXE_CS
]
# Add best model-specific keys using list comprehension
OUTPUT_CS.extend(["best_model"])
OUTPUT_CS += [f"best_{suffix}" for suffix in MODEL_SUFFIXES_CS]
# Add other specific keys directly
OUTPUT_CS.extend(
    [
        "median_quantity",
        "median_price",
        "last_price",
        "last_date",
        "quality_test",
        "quality_test_high",
        "quality_test_medium",
        "elasticity_level",
        "details",
    ]
)
