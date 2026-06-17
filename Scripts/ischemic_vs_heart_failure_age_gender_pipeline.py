"""CDC WONDER ischemic heart disease vs. heart failure analysis.

The script cleans the age/sex/state mortality exports, fits one-breakpoint
segmented regressions, and writes CSV/plot outputs for review in Tableau.
"""

from __future__ import annotations

import math
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.api as sm
from statsmodels.stats.anova import anova_lm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
PLOTS_DIR = PROJECT_ROOT / "plots"

# Input and output files
CLEAN_OUTPUT = DATA_DIR / "ischemic_vs_heart_failure_age_gender_1999_2025_clean.csv"
SUMMARY_OUTPUT = DATA_DIR / "ischemic_vs_heart_failure_age_gender_changepoint_summary.csv"
COMPLETE_SUMMARY_OUTPUT = DATA_DIR / "ischemic_vs_heart_failure_age_gender_changepoint_summary_complete.csv"
VALIDATION_OUTPUT = DATA_DIR / "ischemic_vs_heart_failure_statsmodels_vs_linalg_validation.csv"
DIVERGENCE_OUTPUT = DATA_DIR / "ischemic_vs_heart_failure_age_gender_significant_divergence_table.csv"
MODEL_COMPARISON_OUTPUT = DATA_DIR / "model_comparison_linear_vs_segmented.csv"
MODEL_EVIDENCE_COUNTS_OUTPUT = DATA_DIR / "model_comparison_evidence_counts_by_disease_age.csv"
MODEL_STRONGEST_OUTPUT = DATA_DIR / "model_comparison_top20_strongest_segmented_evidence.csv"
MODEL_WEAKEST_OUTPUT = DATA_DIR / "model_comparison_top20_weakest_segmented_evidence.csv"
ANOVA_COMPARISON_OUTPUT = DATA_DIR / "anova_linear_vs_segmented.csv"
ANOVA_COUNTS_OUTPUT = DATA_DIR / "anova_significance_counts_by_disease_age.csv"
ANOVA_TOP20_OUTPUT = DATA_DIR / "anova_top20_strongest_segmented_improvement.csv"
B2_SIGNIFICANCE_OUTPUT = DATA_DIR / "b2_significance_summary.csv"
B2_COUNTS_BY_DISEASE_OUTPUT = DATA_DIR / "b2_significance_counts_by_disease_category.csv"
B2_COUNTS_BY_AGE_OUTPUT = DATA_DIR / "b2_significance_counts_by_age_group.csv"
B2_COUNTS_BY_DISEASE_AGE_OUTPUT = DATA_DIR / "b2_significance_counts_by_disease_category_age_group.csv"
B2_TOP25_MOST_SIGNIFICANT_OUTPUT = DATA_DIR / "b2_top25_most_significant.csv"
B2_TOP25_LEAST_SIGNIFICANT_OUTPUT = DATA_DIR / "b2_top25_least_significant.csv"
SLOPE_AFTER_SIGNIFICANCE_OUTPUT = DATA_DIR / "slope_after_significance.csv"
SLOPE_AFTER_COUNTS_BY_DISEASE_OUTPUT = DATA_DIR / "slope_after_significance_counts_by_disease_category.csv"
SLOPE_AFTER_COUNTS_BY_AGE_OUTPUT = DATA_DIR / "slope_after_significance_counts_by_age_group.csv"
SLOPE_AFTER_COUNTS_BY_DISEASE_AGE_OUTPUT = DATA_DIR / "slope_after_significance_counts_by_disease_category_age_group.csv"
SLOPE_AFTER_TOP25_DECREASING_OUTPUT = DATA_DIR / "slope_after_top25_most_significantly_decreasing.csv"
SLOPE_AFTER_TOP25_INCREASING_OUTPUT = DATA_DIR / "slope_after_top25_most_significantly_increasing.csv"
TABLEAU_MASTER_OUTPUT = DATA_DIR / "tableau_segmented_regression_master.csv"

# Raw CDC WONDER exports are split at 2020/2021.
DISEASE_FILES = {
    "ischemic": [
        DATA_DIR / "ischemic_mortality_age_groups_1999_2020.csv",
        DATA_DIR / "ischemic_mortality_age_groups_2021_2025.csv",
    ],
    "heart_failure": [
        DATA_DIR / "failure_mortality_age_groups_1999_2020.csv",
        DATA_DIR / "failure_mortality_age_groups_2021_2025.csv",
    ],
}

# Keep the oldest age band separate because the 85+ trend can behave differently
# from the 75-84 group.
AGE_GROUPS = ["55-64 years", "65-74 years", "75-84 years", "85+ years"]
AGE_SLUGS = {
    "55-64 years": "55_64",
    "65-74 years": "65_74",
    "75-84 years": "75_84",
    "85+ years": "85_plus",
}

PLOT_STATES_BY_ABBR = {
    "NJ": "New Jersey",
    "NY": "New York",
    "CA": "California",
    "FL": "Florida",
    "OH": "Ohio",
    "OK": "Oklahoma",
    "NM": "New Mexico",
    "WA": "Washington",
    "TX": "Texas",
}

GENDER_ORDER = ["Male", "Female", "Total"]
COLORS = {"Male": "#c0392b", "Female": "#2878b5", "Total": "#222222"}
BREAKPOINT_LABEL_Y = {"Male": 0.96, "Female": 0.88, "Total": 0.80}

# Basic model inclusion rules.
MIN_YEARS_FOR_MODEL = 10
MIN_CALENDAR_YEARS_PER_SEGMENT = 4
MIN_OBSERVED_YEARS_PER_SEGMENT = 4


# -----------------------------
# Cleaning helpers
# -----------------------------

def slugify(value: object) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def to_snake_case(value: object) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def clean_text(value: object) -> object:
    if pd.isna(value):
        return value
    return re.sub(r"\s+", " ", str(value).strip())


def normalize_gender(value: object) -> object:
    value = clean_text(value)
    if pd.isna(value):
        return value
    key = str(value).lower()
    if key in {"m", "male"}:
        return "Male"
    if key in {"f", "female"}:
        return "Female"
    if key in {"total", "both", "all"}:
        return "Total"
    return str(value).title()


def parse_year(value: object) -> float:
    if pd.isna(value):
        return np.nan
    match = re.search(r"\d{4}", str(value))
    return float(match.group(0)) if match else np.nan


def parse_numeric(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype("string")
        .str.strip()
        .str.replace(",", "", regex=False)
        .str.replace("Unreliable", "", regex=False)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def days_in_year(year: int) -> int:
    year = int(year)
    return 366 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) else 365


def standardize_wonder_columns(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df.columns = [to_snake_case(column) for column in df.columns]
    return df.rename(
        columns={
            "residence_state": "state",
            "residence_state_code": "state_code",
            "ten_year_age_groups": "age_group",
            "ten_year_age_groups_code": "age_group_code",
            "sex": "gender",
        }
    )


def clean_wonder_file(path: Path, disease_category: str) -> pd.DataFrame:
    raw = pd.read_csv(path)
    df = standardize_wonder_columns(raw)

    required = ["state", "year", "age_group", "gender", "deaths", "population"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {missing}")

    if "crude_rate" not in df.columns:
        df["crude_rate"] = np.nan

    df = df[[*required, "crude_rate"]].copy()
    for column in ["state", "age_group", "gender"]:
        df[column] = df[column].map(clean_text)

    df["gender"] = df["gender"].map(normalize_gender)
    df["year"] = df["year"].map(parse_year)

    # Suppressed/non-numeric death counts are missing rather than zero.
    df["deaths"] = parse_numeric(df["deaths"])
    df["population"] = parse_numeric(df["population"])
    df["crude_rate"] = parse_numeric(df["crude_rate"])

    df = df[df["age_group"].isin(AGE_GROUPS)].copy()
    df = df.dropna(subset=["state", "year", "age_group", "gender", "population"])
    df = df[df["population"] > 0].copy()

    df["year"] = df["year"].astype(int)
    df["days_in_year"] = df["year"].map(days_in_year)
    df["disease_category"] = disease_category
    df["source_file"] = path.name

    df["average_daily_crude_rate"] = (
        df["deaths"] / df["population"] / df["days_in_year"] * 100_000
    )

    return df[
        [
            "disease_category",
            "state",
            "year",
            "age_group",
            "gender",
            "deaths",
            "population",
            "crude_rate",
            "days_in_year",
            "average_daily_crude_rate",
            "source_file",
        ]
    ]


def add_total_gender_rows(gender_df: pd.DataFrame) -> pd.DataFrame:
    parts = []
    group_columns = ["disease_category", "state", "year", "age_group"]
    for keys, group in gender_df.groupby(group_columns, dropna=False):
        total = dict(zip(group_columns, keys))
        total["gender"] = "Total"

        has_suppressed_deaths = group["deaths"].isna().any()
        total["deaths"] = np.nan if has_suppressed_deaths else group["deaths"].sum()
        total["population"] = group["population"].sum()
        total["days_in_year"] = int(group["days_in_year"].iloc[0])
        total["crude_rate"] = (
            np.nan
            if pd.isna(total["deaths"]) or total["population"] <= 0
            else total["deaths"] / total["population"] * 100_000
        )
        total["average_daily_crude_rate"] = (
            np.nan
            if pd.isna(total["deaths"]) or total["population"] <= 0
            else total["deaths"] / total["population"] / total["days_in_year"] * 100_000
        )
        total["source_file"] = "computed_total_from_clean_gender_rows"
        parts.append(total)

    total_df = pd.DataFrame(parts)
    return pd.concat([gender_df, total_df], ignore_index=True, sort=False)


def load_clean_dataset() -> pd.DataFrame:
    cleaned_parts = []
    for disease_category, paths in DISEASE_FILES.items():
        for path in paths:
            if not path.exists():
                raise FileNotFoundError(path)
            cleaned_parts.append(clean_wonder_file(path, disease_category))

    gender_df = pd.concat(cleaned_parts, ignore_index=True, sort=False)
    duplicate_key = ["state", "year", "age_group", "gender", "disease_category"]
    gender_df = gender_df.drop_duplicates(subset=duplicate_key, keep="last")
    gender_df = add_total_gender_rows(gender_df)
    gender_df = gender_df.drop_duplicates(subset=duplicate_key, keep="last")
    gender_df = gender_df.sort_values(
        ["disease_category", "age_group", "state", "gender", "year"]
    ).reset_index(drop=True)
    return gender_df


# -----------------------------
# Regression helpers
# -----------------------------

def design_matrix(years: np.ndarray, changepoint_year: int) -> tuple[np.ndarray, float]:
    first_year = float(years.min())
    x = years - first_year
    hinge = np.maximum(0, years - changepoint_year)
    matrix = np.column_stack([np.ones(len(years)), x, hinge])
    return matrix, first_year


def reversed_hinge_design_matrix(
    years: np.ndarray, changepoint_year: int
) -> tuple[np.ndarray, float]:
    first_year = float(years.min())
    x = years - first_year
    reversed_hinge = np.maximum(0, changepoint_year - years)
    matrix = np.column_stack([np.ones(len(years)), x, reversed_hinge])
    return matrix, first_year


def prepare_model_data(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df[["year", "average_daily_crude_rate"]]
        .dropna()
        .drop_duplicates(subset=["year"], keep="last")
        .sort_values("year")
    )


def segment_observation_counts(df: pd.DataFrame, changepoint_year: int) -> tuple[int, int]:
    pre_count = int((df["year"] <= changepoint_year).sum())
    post_count = int((df["year"] > changepoint_year).sum())
    return pre_count, post_count


def regression_metrics(
    observed: np.ndarray, predicted: np.ndarray, parameter_count: int
) -> dict[str, float]:
    n = len(observed)
    residuals = observed - predicted
    sse = float(np.sum(residuals**2))
    safe_mse = max(sse, np.finfo(float).eps) / n
    aic = float(n * math.log(safe_mse) + 2 * parameter_count)
    bic = float(n * math.log(safe_mse) + parameter_count * math.log(n))

    tss = float(np.sum((observed - observed.mean()) ** 2))
    if tss <= np.finfo(float).eps:
        r_squared = 1.0 if sse <= np.finfo(float).eps else np.nan
    else:
        r_squared = 1 - (sse / tss)

    if pd.isna(r_squared) or n <= parameter_count:
        adj_r2 = np.nan
    else:
        adj_r2 = 1 - (1 - r_squared) * (n - 1) / (n - parameter_count)

    return {
        "sse": sse,
        "aic": aic,
        "bic": bic,
        "adj_r2": float(adj_r2) if not pd.isna(adj_r2) else np.nan,
    }


def fit_linear_regression(df: pd.DataFrame) -> dict[str, object] | None:
    df = prepare_model_data(df)
    if len(df) < 3:
        return None

    years = df["year"].to_numpy(dtype=float)
    rates = df["average_daily_crude_rate"].to_numpy(dtype=float)
    first_year = float(years.min())
    x = years - first_year
    matrix = np.column_stack([np.ones(len(x)), x])
    model = sm.OLS(rates, matrix).fit()
    predicted = model.predict(matrix)
    metrics = regression_metrics(rates, predicted, matrix.shape[1])

    return {
        "linear_intercept": float(model.params[0]),
        "linear_slope": float(model.params[1]),
        "n_years_used": int(len(df)),
        "first_year": first_year,
        **metrics,
    }


def fit_linear_ols_result(df: pd.DataFrame):
    df = prepare_model_data(df)
    years = df["year"].to_numpy(dtype=float)
    rates = df["average_daily_crude_rate"].to_numpy(dtype=float)
    first_year = float(years.min())
    x = years - first_year
    matrix = np.column_stack([np.ones(len(x)), x])
    return sm.OLS(rates, matrix).fit()


def fit_segmented_ols_result(df: pd.DataFrame, changepoint_year: int):
    df = prepare_model_data(df)
    years = df["year"].to_numpy(dtype=float)
    rates = df["average_daily_crude_rate"].to_numpy(dtype=float)
    matrix, _ = design_matrix(years, changepoint_year)
    return sm.OLS(rates, matrix).fit()


def fit_for_changepoint(df: pd.DataFrame, changepoint_year: int) -> dict[str, object]:
    years = df["year"].to_numpy(dtype=float)
    rates = df["average_daily_crude_rate"].to_numpy(dtype=float)
    matrix, first_year = design_matrix(years, changepoint_year)
    model = sm.OLS(rates, matrix).fit()

    beta0, beta1, beta2 = model.params
    predicted = model.predict(matrix)
    metrics = regression_metrics(rates, predicted, matrix.shape[1])
    slope_after_test = model.t_test([0, 1, 1])
    pre_count, post_count = segment_observation_counts(df, changepoint_year)

    return {
        "changepoint_year": int(changepoint_year),
        "intercept": float(beta0),
        "b0": float(beta0),
        "b1": float(beta1),
        "b2": float(beta2),
        "standard_error_b2": float(model.bse[2]),
        "t_stat_b2": float(model.tvalues[2]),
        "p_value_b2": float(model.pvalues[2]),
        "significance_b2": bool(model.pvalues[2] < 0.05),
        "b2_significance": classify_p_value(model.pvalues[2]),
        "slope_before": float(beta1),
        "change_in_slope": float(beta2),
        "slope_after": float(beta1 + beta2),
        "slope_after_p_value": float(slope_after_test.pvalue),
        "sse": metrics["sse"],
        "aic": metrics["aic"],
        "bic": metrics["bic"],
        "adj_r2": metrics["adj_r2"],
        "n_years_used": int(len(rates)),
        "observed_years_before_changepoint": pre_count,
        "observed_years_after_changepoint": post_count,
        "sparse_segment_flag": bool(
            min(pre_count, post_count) < MIN_OBSERVED_YEARS_PER_SEGMENT
        ),
        "first_year": float(first_year),
        "statsmodels_params": model.params,
        "predicted": predicted,
        "matrix": matrix,
    }


def classify_slope(slope_after: float, p_value: float) -> str:
    if pd.isna(slope_after) or pd.isna(p_value):
        return "no_significant_change"
    if slope_after > 0 and p_value < 0.05:
        return "significant_increase"
    if slope_after < 0 and p_value < 0.05:
        return "significant_decrease"
    return "no_significant_change"


def continuous_segmented_regression(df: pd.DataFrame) -> dict[str, object] | None:
    df = prepare_model_data(df)
    if len(df) < MIN_YEARS_FOR_MODEL:
        return None

    min_year = int(df["year"].min() + MIN_CALENDAR_YEARS_PER_SEGMENT)
    max_year = int(df["year"].max() - MIN_CALENDAR_YEARS_PER_SEGMENT)
    if min_year > max_year:
        return None

    best: dict[str, object] | None = None
    for changepoint_year in range(min_year, max_year + 1):
        fitted = fit_for_changepoint(df, changepoint_year)
        if best is None or fitted["sse"] < best["sse"]:
            best = fitted

    if best is None:
        return None

    matrix = best.pop("matrix")
    rates = df["average_daily_crude_rate"].to_numpy(dtype=float)
    linalg_params, *_ = np.linalg.lstsq(matrix, rates, rcond=None)
    linalg_predicted = matrix @ linalg_params
    best["linalg_intercept"] = float(linalg_params[0])
    best["linalg_slope_before"] = float(linalg_params[1])
    best["linalg_change_in_slope"] = float(linalg_params[2])
    best["linalg_slope_after"] = float(linalg_params[1] + linalg_params[2])
    best["linalg_sse"] = float(np.sum((rates - linalg_predicted) ** 2))
    return best


def run_segmented_regressions(clean_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    results = []
    validation = []
    group_columns = ["disease_category", "age_group", "state", "gender"]

    for keys, subset in clean_df.groupby(group_columns, dropna=False):
        result = continuous_segmented_regression(subset)
        if result is None:
            continue

        disease_category, age_group, state, gender = keys
        result_row = {
            "disease_category": disease_category,
            "age_group": age_group,
            "state": state,
            "gender": gender,
            "changepoint_year": result["changepoint_year"],
            "b0": result["b0"],
            "b1": result["b1"],
            "b2": result["b2"],
            "standard_error_b2": result["standard_error_b2"],
            "t_stat_b2": result["t_stat_b2"],
            "p_value_b2": result["p_value_b2"],
            "significance_b2": result["significance_b2"],
            "b2_significance": result["b2_significance"],
            "slope_before": result["slope_before"],
            "slope_after": result["slope_after"],
            "change_in_slope": result["change_in_slope"],
            "slope_after_p_value": result["slope_after_p_value"],
            "significance_label": classify_slope(
                result["slope_after"], result["slope_after_p_value"]
            ),
            "sse": result["sse"],
            "aic": result["aic"],
            "n_years_used": result["n_years_used"],
            "observed_years_before_changepoint": result[
                "observed_years_before_changepoint"
            ],
            "observed_years_after_changepoint": result[
                "observed_years_after_changepoint"
            ],
            "sparse_segment_flag": result["sparse_segment_flag"],
            "intercept": result["intercept"],
            "first_year": result["first_year"],
        }
        results.append(result_row)

        validation.append(
            {
                "disease_category": disease_category,
                "age_group": age_group,
                "state": state,
                "gender": gender,
                "changepoint_year": result["changepoint_year"],
                "intercept_abs_diff": abs(result["intercept"] - result["linalg_intercept"]),
                "slope_before_abs_diff": abs(
                    result["slope_before"] - result["linalg_slope_before"]
                ),
                "change_in_slope_abs_diff": abs(
                    result["change_in_slope"] - result["linalg_change_in_slope"]
                ),
                "slope_after_abs_diff": abs(
                    result["slope_after"] - result["linalg_slope_after"]
                ),
                "sse_abs_diff": abs(result["sse"] - result["linalg_sse"]),
            }
        )

    results_df = pd.DataFrame(results)
    validation_df = pd.DataFrame(validation)
    if results_df.empty:
        raise ValueError("No segmented regression results were produced.")

    last_5_years = (
        clean_df[clean_df["year"] >= clean_df["year"].max() - 4]
        .groupby(group_columns, as_index=False)["average_daily_crude_rate"]
        .mean()
        .rename(columns={"average_daily_crude_rate": "mean_rate_last_5y"})
    )

    summary_df = results_df.merge(last_5_years, on=group_columns, how="left")
    ordered_columns = [
        "disease_category",
        "age_group",
        "state",
        "gender",
        "changepoint_year",
        "b0",
        "b1",
        "b2",
        "standard_error_b2",
        "t_stat_b2",
        "p_value_b2",
        "significance_b2",
        "b2_significance",
        "slope_before",
        "slope_after",
        "change_in_slope",
        "slope_after_p_value",
        "significance_label",
        "mean_rate_last_5y",
        "sse",
        "aic",
        "n_years_used",
        "observed_years_before_changepoint",
        "observed_years_after_changepoint",
        "sparse_segment_flag",
        "intercept",
        "first_year",
    ]
    summary_df = summary_df[ordered_columns].sort_values(group_columns).reset_index(drop=True)
    return summary_df, validation_df


# -----------------------------
# Model comparison / summaries
# -----------------------------

def classify_model_evidence(delta_aic: float) -> str:
    if delta_aic <= -10:
        return "Strong segmented evidence"
    if delta_aic <= -4:
        return "Moderate segmented evidence"
    return "Weak/no segmented evidence"


def run_model_comparison(clean_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_columns = ["state", "disease_category", "age_group", "gender"]

    for keys, subset in clean_df.groupby(group_columns, dropna=False):
        state, disease_category, age_group, gender = keys
        model_df = prepare_model_data(subset)

        if len(model_df) < MIN_YEARS_FOR_MODEL:
            continue

        linear = fit_linear_regression(model_df)
        segmented = continuous_segmented_regression(model_df)
        if linear is None or segmented is None:
            continue

        delta_aic = segmented["aic"] - linear["aic"]
        delta_bic = segmented["bic"] - linear["bic"]
        percent_sse_reduction = (
            np.nan
            if linear["sse"] <= np.finfo(float).eps
            else (linear["sse"] - segmented["sse"]) / linear["sse"] * 100
        )

        rows.append(
            {
                "state": state,
                "disease_category": disease_category,
                "age_group": age_group,
                "gender": gender,
                "n_years_used": linear["n_years_used"],
                "linear_slope": linear["linear_slope"],
                "linear_sse": linear["sse"],
                "linear_aic": linear["aic"],
                "linear_bic": linear["bic"],
                "linear_adj_r2": linear["adj_r2"],
                "segmented_changepoint_year": segmented["changepoint_year"],
                "segmented_slope_before": segmented["slope_before"],
                "segmented_slope_after": segmented["slope_after"],
                "segmented_sse": segmented["sse"],
                "segmented_aic": segmented["aic"],
                "segmented_bic": segmented["bic"],
                "segmented_adj_r2": segmented["adj_r2"],
                "delta_aic": delta_aic,
                "delta_bic": delta_bic,
                "percent_sse_reduction": percent_sse_reduction,
                "model_evidence": classify_model_evidence(delta_aic),
            }
        )

    if not rows:
        raise ValueError("No model comparison rows were produced.")

    ordered_columns = [
        "state",
        "disease_category",
        "age_group",
        "gender",
        "n_years_used",
        "linear_slope",
        "linear_sse",
        "linear_aic",
        "linear_bic",
        "linear_adj_r2",
        "segmented_changepoint_year",
        "segmented_slope_before",
        "segmented_slope_after",
        "segmented_sse",
        "segmented_aic",
        "segmented_bic",
        "segmented_adj_r2",
        "delta_aic",
        "delta_bic",
        "percent_sse_reduction",
        "model_evidence",
    ]
    return pd.DataFrame(rows)[ordered_columns].sort_values(group_columns).reset_index(drop=True)


def write_model_comparison_qa(model_comparison_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    evidence_counts = (
        model_comparison_df.groupby(
            ["disease_category", "age_group", "model_evidence"], as_index=False
        )
        .size()
        .rename(columns={"size": "n"})
        .sort_values(["disease_category", "age_group", "model_evidence"])
    )
    strongest = model_comparison_df.sort_values("delta_aic").head(20)
    weakest = model_comparison_df.sort_values("delta_aic", ascending=False).head(20)

    evidence_counts.to_csv(MODEL_EVIDENCE_COUNTS_OUTPUT, index=False)
    strongest.to_csv(MODEL_STRONGEST_OUTPUT, index=False)
    weakest.to_csv(MODEL_WEAKEST_OUTPUT, index=False)
    return evidence_counts, strongest, weakest


def classify_p_value(p_value: float) -> str:
    if pd.isna(p_value):
        return "not_testable"
    if p_value < 0.001:
        return "p < 0.001"
    if p_value < 0.01:
        return "p < 0.01"
    if p_value < 0.05:
        return "p < 0.05"
    return "not_significant"


def classify_anova_p_value(p_value: float) -> str:
    return classify_p_value(p_value)


def run_anova_model_comparison(
    clean_df: pd.DataFrame, model_comparison_df: pd.DataFrame
) -> pd.DataFrame:
    rows = []

    for _, row in model_comparison_df.iterrows():
        subset = clean_df[
            (clean_df["state"] == row["state"])
            & (clean_df["disease_category"] == row["disease_category"])
            & (clean_df["age_group"] == row["age_group"])
            & (clean_df["gender"] == row["gender"])
        ]
        n = int(row["n_years_used"])

        try:
            linear_model = fit_linear_ols_result(subset)
            segmented_model = fit_segmented_ols_result(
                subset, int(row["segmented_changepoint_year"])
            )
            anova_table = anova_lm(linear_model, segmented_model)
            linear_anova = anova_table.iloc[0]
            segmented_anova = anova_table.iloc[1]
            linear_df_resid = float(linear_anova["df_resid"])
            segmented_df_resid = float(segmented_anova["df_resid"])
            df_diff = float(segmented_anova["df_diff"])
            ss_diff = float(segmented_anova["ss_diff"])
            f_statistic = float(segmented_anova["F"])
            p_value = float(segmented_anova["Pr(>F)"])
            linear_ssr = float(linear_anova["ssr"])
            segmented_ssr = float(segmented_anova["ssr"])
        except Exception:
            linear_df_resid = np.nan
            segmented_df_resid = np.nan
            df_diff = np.nan
            ss_diff = np.nan
            f_statistic = np.nan
            p_value = np.nan
            linear_ssr = row["linear_sse"]
            segmented_ssr = row["segmented_sse"]

        rows.append(
            {
                "state": row["state"],
                "disease_category": row["disease_category"],
                "age_group": row["age_group"],
                "gender": row["gender"],
                "n_years_used": n,
                "linear_df_resid": linear_df_resid,
                "segmented_df_resid": segmented_df_resid,
                "df_diff": df_diff,
                "linear_ssr": linear_ssr,
                "segmented_ssr": segmented_ssr,
                "ss_diff": ss_diff,
                "f_statistic": f_statistic,
                "anova_p_value": p_value,
                "anova_significance": classify_anova_p_value(p_value),
                "delta_aic": row["delta_aic"],
                "delta_bic": row["delta_bic"],
                "percent_sse_reduction": row["percent_sse_reduction"],
                "model_evidence": row["model_evidence"],
                "segmented_changepoint_year": row["segmented_changepoint_year"],
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["disease_category", "age_group", "state", "gender"]
    ).reset_index(drop=True)


def write_anova_qa(anova_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    counts = (
        anova_df.groupby(
            ["disease_category", "age_group", "anova_significance"], as_index=False
        )
        .size()
        .rename(columns={"size": "n"})
        .sort_values(["disease_category", "age_group", "anova_significance"])
    )
    top20 = anova_df.sort_values("anova_p_value", na_position="last").head(20)
    counts.to_csv(ANOVA_COUNTS_OUTPUT, index=False)
    top20.to_csv(ANOVA_TOP20_OUTPUT, index=False)
    return counts, top20


def add_anova_results_to_summary(
    summary_df: pd.DataFrame, anova_df: pd.DataFrame
) -> pd.DataFrame:
    join_columns = ["state", "disease_category", "age_group", "gender"]
    anova_columns = [
        *join_columns,
        "f_statistic",
        "anova_p_value",
        "anova_significance",
    ]
    anova_for_summary = anova_df[anova_columns].rename(
        columns={"f_statistic": "anova_f_statistic"}
    )
    return summary_df.merge(anova_for_summary, on=join_columns, how="left")


def write_b2_significance_outputs(
    summary_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    b2_summary = summary_df[
        [
            "state",
            "disease_category",
            "age_group",
            "gender",
            "changepoint_year",
            "b1",
            "b2",
            "slope_after",
            "p_value_b2",
            "significance_b2",
            "b2_significance",
        ]
    ].rename(columns={"slope_after": "slope2"})

    counts_by_disease = (
        b2_summary.groupby(["disease_category", "b2_significance"], as_index=False)
        .size()
        .rename(columns={"size": "n"})
        .sort_values(["disease_category", "b2_significance"])
    )
    counts_by_age = (
        b2_summary.groupby(["age_group", "b2_significance"], as_index=False)
        .size()
        .rename(columns={"size": "n"})
        .sort_values(["age_group", "b2_significance"])
    )
    counts_by_disease_age = (
        b2_summary.groupby(
            ["disease_category", "age_group", "b2_significance"], as_index=False
        )
        .size()
        .rename(columns={"size": "n"})
        .sort_values(["disease_category", "age_group", "b2_significance"])
    )
    top25_most = b2_summary.sort_values("p_value_b2", na_position="last").head(25)
    top25_least = b2_summary.sort_values(
        "p_value_b2", ascending=False, na_position="last"
    ).head(25)

    b2_summary.to_csv(B2_SIGNIFICANCE_OUTPUT, index=False)
    counts_by_disease.to_csv(B2_COUNTS_BY_DISEASE_OUTPUT, index=False)
    counts_by_age.to_csv(B2_COUNTS_BY_AGE_OUTPUT, index=False)
    counts_by_disease_age.to_csv(B2_COUNTS_BY_DISEASE_AGE_OUTPUT, index=False)
    top25_most.to_csv(B2_TOP25_MOST_SIGNIFICANT_OUTPUT, index=False)
    top25_least.to_csv(B2_TOP25_LEAST_SIGNIFICANT_OUTPUT, index=False)
    return (
        b2_summary,
        counts_by_disease,
        counts_by_age,
        counts_by_disease_age,
        top25_most,
        top25_least,
    )


def write_complete_changepoint_summary(clean_df: pd.DataFrame, summary_df: pd.DataFrame) -> pd.DataFrame:
    states = sorted(clean_df["state"].dropna().unique())
    disease_categories = sorted(clean_df["disease_category"].dropna().unique())
    expected = pd.MultiIndex.from_product(
        [states, disease_categories, AGE_GROUPS, GENDER_ORDER],
        names=["state", "disease_category", "age_group", "gender"],
    ).to_frame(index=False)

    usable_years = (
        clean_df.dropna(subset=["average_daily_crude_rate"])
        .groupby(["state", "disease_category", "age_group", "gender"], as_index=False)
        .agg(
            usable_years=("year", "nunique"),
            usable_min_year=("year", "min"),
            usable_max_year=("year", "max"),
        )
    )

    complete = expected.merge(
        summary_df, on=["state", "disease_category", "age_group", "gender"], how="left"
    ).merge(
        usable_years, on=["state", "disease_category", "age_group", "gender"], how="left"
    )
    complete["usable_years"] = complete["usable_years"].fillna(0).astype(int)

    is_fitted = complete["changepoint_year"].notna()
    complete["model_status"] = np.where(is_fitted, "fitted", "not_testable")
    complete["exclusion_reason"] = np.where(
        is_fitted, "", "fewer_than_10_usable_years"
    )

    for column in ["b2_significance", "anova_significance", "significance_label"]:
        if column in complete.columns:
            complete[column] = complete[column].fillna("not_testable")

    complete.to_csv(COMPLETE_SUMMARY_OUTPUT, index=False)
    return complete


def fit_reversed_hinge_model(df: pd.DataFrame, changepoint_year: int) -> dict[str, float]:
    df = prepare_model_data(df)
    years = df["year"].to_numpy(dtype=float)
    rates = df["average_daily_crude_rate"].to_numpy(dtype=float)
    matrix, _ = reversed_hinge_design_matrix(years, changepoint_year)
    model = sm.OLS(rates, matrix).fit()
    beta0, beta1, beta2 = model.params
    return {
        "b0": float(beta0),
        "b1": float(beta1),
        "b1_p_value": float(model.pvalues[1]),
        "b2": float(beta2),
        "b2_p_value": float(model.pvalues[2]),
    }


def run_reversed_hinge_slope_after_significance(
    clean_df: pd.DataFrame, summary_df: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    join_columns = ["state", "disease_category", "age_group", "gender"]

    for _, summary_row in summary_df.iterrows():
        subset = clean_df[
            (clean_df["state"] == summary_row["state"])
            & (clean_df["disease_category"] == summary_row["disease_category"])
            & (clean_df["age_group"] == summary_row["age_group"])
            & (clean_df["gender"] == summary_row["gender"])
        ]
        changepoint_year = int(summary_row["changepoint_year"])
        fit = fit_reversed_hinge_model(subset, changepoint_year)
        slope_after_diff = abs(fit["b1"] - float(summary_row["slope_after"]))

        rows.append(
            {
                "state": summary_row["state"],
                "disease_category": summary_row["disease_category"],
                "age_group": summary_row["age_group"],
                "gender": summary_row["gender"],
                "changepoint_year": changepoint_year,
                "b0": fit["b0"],
                "b1": fit["b1"],
                "b1_p_value": fit["b1_p_value"],
                "slope_after_significant": bool(fit["b1_p_value"] < 0.05),
                "slope_after_significance": classify_p_value(fit["b1_p_value"]),
                "b2": fit["b2"],
                "b2_p_value": fit["b2_p_value"],
                "original_b1_slope_before": summary_row["b1"],
                "original_b2": summary_row["b2"],
                "b2_abs_diff": abs(fit["b2"] - float(summary_row["b2"])),
                "mean_rate_last_5y": summary_row["mean_rate_last_5y"],
                "original_slope_after": summary_row["slope_after"],
                "slope_after_abs_diff": slope_after_diff,
            }
        )

    result = pd.DataFrame(rows).sort_values(join_columns).reset_index(drop=True)
    max_diff = result["slope_after_abs_diff"].max()
    if max_diff > 1e-10:
        raise ValueError(
            "Reversed-hinge b1 did not match original slope_after. "
            f"Maximum absolute difference: {max_diff}"
        )

    export_columns = [
        "state",
        "disease_category",
        "age_group",
        "gender",
        "changepoint_year",
        "b0",
        "b1",
        "b1_p_value",
        "slope_after_significant",
        "slope_after_significance",
        "b2",
        "b2_p_value",
        "original_b1_slope_before",
        "original_b2",
        "b2_abs_diff",
        "mean_rate_last_5y",
        "original_slope_after",
        "slope_after_abs_diff",
    ]
    return result[export_columns]


def write_slope_after_significance_outputs(
    slope_after_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    counts_by_disease = (
        slope_after_df.groupby(
            ["disease_category", "slope_after_significant"], as_index=False
        )
        .size()
        .rename(columns={"size": "n"})
        .sort_values(["disease_category", "slope_after_significant"])
    )
    counts_by_age = (
        slope_after_df.groupby(["age_group", "slope_after_significant"], as_index=False)
        .size()
        .rename(columns={"size": "n"})
        .sort_values(["age_group", "slope_after_significant"])
    )
    counts_by_disease_age = (
        slope_after_df.groupby(
            ["disease_category", "age_group", "slope_after_significant"], as_index=False
        )
        .size()
        .rename(columns={"size": "n"})
        .sort_values(["disease_category", "age_group", "slope_after_significant"])
    )
    top25_decreasing = (
        slope_after_df[slope_after_df["b1"] < 0]
        .sort_values(["b1_p_value", "b1"], ascending=[True, True])
        .head(25)
    )
    top25_increasing = (
        slope_after_df[slope_after_df["b1"] > 0]
        .sort_values(["b1_p_value", "b1"], ascending=[True, False])
        .head(25)
    )

    slope_after_df.to_csv(SLOPE_AFTER_SIGNIFICANCE_OUTPUT, index=False)
    counts_by_disease.to_csv(SLOPE_AFTER_COUNTS_BY_DISEASE_OUTPUT, index=False)
    counts_by_age.to_csv(SLOPE_AFTER_COUNTS_BY_AGE_OUTPUT, index=False)
    counts_by_disease_age.to_csv(SLOPE_AFTER_COUNTS_BY_DISEASE_AGE_OUTPUT, index=False)
    top25_decreasing.to_csv(SLOPE_AFTER_TOP25_DECREASING_OUTPUT, index=False)
    top25_increasing.to_csv(SLOPE_AFTER_TOP25_INCREASING_OUTPUT, index=False)
    return (
        counts_by_disease,
        counts_by_age,
        counts_by_disease_age,
        top25_decreasing,
        top25_increasing,
    )


def classify_post_break_direction(slope_after: float, p_value: float) -> str:
    if pd.isna(slope_after) or pd.isna(p_value):
        return "not_testable"
    if p_value >= 0.05:
        return "not_significant"
    return "rising" if slope_after > 0 else "falling"


def classify_trend_change(row: pd.Series) -> str:
    if row.get("model_status") != "fitted" or pd.isna(row.get("p_value_b2")):
        return "not_testable"
    if row["p_value_b2"] >= 0.05:
        return "no_significant_slope_change"
    return "slope_increased" if row["b2"] > 0 else "slope_decreased"


def classify_priority(row: pd.Series) -> str:
    if row.get("model_status") != "fitted":
        return "not_testable"
    if bool(row.get("sparse_segment_flag", False)):
        return "sparse_segment_review"
    if (
        row.get("post_break_direction") == "rising"
        and row.get("burden_percentile_within_disease_age", 0) >= 0.75
    ):
        return "high_burden_rising"
    if row.get("b2_significance") in {"p < 0.001", "p < 0.01", "p < 0.05"}:
        return "significant_slope_change"
    return "lower_evidence"


def write_tableau_master(
    complete_summary_df: pd.DataFrame,
    slope_after_df: pd.DataFrame,
    model_comparison_df: pd.DataFrame,
) -> pd.DataFrame:
    join_columns = ["state", "disease_category", "age_group", "gender"]

    slope_after_for_master = slope_after_df[
        [
            *join_columns,
            "b1_p_value",
            "slope_after_significant",
            "slope_after_significance",
            "original_b1_slope_before",
            "original_b2",
            "b2_abs_diff",
            "original_slope_after",
            "slope_after_abs_diff",
        ]
    ].rename(
        columns={
            "b1_p_value": "reversed_hinge_slope_after_p_value",
            "slope_after_significant": "reversed_hinge_slope_after_significant",
            "slope_after_significance": "reversed_hinge_slope_after_significance",
        }
    )

    comparison_for_master = model_comparison_df[
        [
            *join_columns,
            "linear_slope",
            "linear_adj_r2",
            "segmented_adj_r2",
            "delta_aic",
            "delta_bic",
            "percent_sse_reduction",
            "model_evidence",
        ]
    ]

    master = (
        complete_summary_df.merge(slope_after_for_master, on=join_columns, how="left")
        .merge(comparison_for_master, on=join_columns, how="left")
        .sort_values(join_columns)
        .reset_index(drop=True)
    )

    master["sparse_segment_flag"] = master["sparse_segment_flag"].fillna(False)
    master["post_break_direction"] = master.apply(
        lambda row: classify_post_break_direction(
            row.get("slope_after"), row.get("slope_after_p_value")
        ),
        axis=1,
    )
    master["trend_change_type"] = master.apply(classify_trend_change, axis=1)
    master["burden_percentile_within_disease_age"] = master.groupby(
        ["disease_category", "age_group"]
    )["mean_rate_last_5y"].rank(pct=True)
    master["priority_group"] = master.apply(classify_priority, axis=1)

    master.to_csv(TABLEAU_MASTER_OUTPUT, index=False)
    return master


# -----------------------------
# Plotting
# -----------------------------

def fitted_values(row: pd.Series, years: np.ndarray) -> np.ndarray:
    x = years - float(row["first_year"])
    hinge = np.maximum(0, years - int(row["changepoint_year"]))
    return (
        float(row["intercept"])
        + float(row["slope_before"]) * x
        + float(row["change_in_slope"]) * hinge
    )


def y_axis_ranges(clean_df: pd.DataFrame) -> dict[tuple[str, str], tuple[float, float]]:
    ranges = {}
    for keys, subset in clean_df.groupby(["disease_category", "age_group"]):
        values = subset["average_daily_crude_rate"].dropna()
        if values.empty:
            continue
        y_min = float(values.min())
        y_max = float(values.max())
        pad = (y_max - y_min) * 0.05 if y_max > y_min else max(y_max * 0.05, 0.01)
        ranges[keys] = (max(0, y_min - pad), y_max + pad)
    return ranges


def plot_segmented_states(clean_df: pd.DataFrame, summary_df: pd.DataFrame) -> list[dict[str, str]]:
    plot_links = []
    ranges = y_axis_ranges(clean_df)

    for disease_category in sorted(clean_df["disease_category"].unique()):
        for age_group in AGE_GROUPS:
            age_slug = AGE_SLUGS[age_group]
            output_dir = PLOTS_DIR / disease_category / age_slug
            output_dir.mkdir(parents=True, exist_ok=True)

            disease_age_df = clean_df[
                (clean_df["disease_category"] == disease_category)
                & (clean_df["age_group"] == age_group)
            ]
            disease_age_results = summary_df[
                (summary_df["disease_category"] == disease_category)
                & (summary_df["age_group"] == age_group)
            ]
            if disease_age_df.empty:
                continue

            y_range = ranges.get((disease_category, age_group))

            for abbr, state in PLOT_STATES_BY_ABBR.items():
                state_df = disease_age_df[disease_age_df["state"] == state]
                if state_df.empty:
                    continue

                fig = go.Figure()
                for gender in GENDER_ORDER:
                    subset = state_df[state_df["gender"] == gender].sort_values("year")
                    reg = disease_age_results[
                        (disease_age_results["state"] == state)
                        & (disease_age_results["gender"] == gender)
                    ]
                    if subset.empty or reg.empty:
                        continue

                    reg_row = reg.iloc[0]
                    years = subset["year"].to_numpy(dtype=int)
                    fit = fitted_values(reg_row, years)
                    color = COLORS[gender]

                    fig.add_trace(
                        go.Scatter(
                            x=years,
                            y=subset["average_daily_crude_rate"],
                            mode="lines+markers",
                            name=f"{gender} observed",
                            line={"color": color, "width": 1.5},
                            marker={"size": 5},
                            opacity=0.45,
                        )
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=years,
                            y=fit,
                            mode="lines",
                            name=f"{gender} fitted",
                            line={"color": color, "width": 3},
                        )
                    )
                    fig.add_vline(
                        x=int(reg_row["changepoint_year"]),
                        line={"color": color, "width": 1, "dash": "dot"},
                    )

                title = (
                    f"{state} ({abbr}): {disease_category.replace('_', ' ').title()} "
                    f"Mortality, {age_group}"
                )
                fig.update_layout(
                    title=title,
                    xaxis_title="Year",
                    yaxis_title="Average daily crude rate per 100,000",
                    yaxis_range=y_range,
                    template="plotly_white",
                    legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
                    margin={"l": 70, "r": 30, "t": 90, "b": 60},
                )

                filename = f"{slugify(state)}_segmented.html"
                fig.write_html(output_dir / filename, include_plotlyjs="cdn")
                plot_links.append(
                    {
                        "disease_category": disease_category,
                        "age_group": age_group,
                        "state": state,
                        "path": f"{disease_category}/{age_slug}/{filename}",
                    }
                )
    return plot_links


def add_breakpoint_annotation(
    fig: go.Figure,
    row: int,
    col: int,
    gender: str,
    breakpoint_year: int,
    y_range: tuple[float, float] | None,
) -> None:
    if y_range is None:
        y_value = 1
    else:
        y_min, y_max = y_range
        y_value = y_min + (y_max - y_min) * BREAKPOINT_LABEL_Y[gender]

    fig.add_annotation(
        x=breakpoint_year,
        y=y_value,
        text=f"{gender} BP: {breakpoint_year}",
        showarrow=True,
        arrowhead=2,
        arrowsize=0.8,
        arrowwidth=1.2,
        arrowcolor=COLORS[gender],
        ax=0,
        ay=-26,
        font={"color": COLORS[gender], "size": 13},
        bgcolor="rgba(255,255,255,0.88)",
        bordercolor=COLORS[gender],
        borderwidth=1,
        borderpad=3,
        row=row,
        col=col,
    )


def write_static_dashboard_png(
    clean_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    disease_category: str,
    state: str,
    output_path: Path,
    ranges: dict[tuple[str, str], tuple[float, float]],
) -> None:
    disease_label = disease_category.replace("_", " ").title()
    fig, axes = plt.subplots(2, 2, figsize=(18, 12), constrained_layout=False)
    fig.suptitle(
        f"{state}: {disease_label} Mortality by Age and Gender",
        fontsize=25,
        fontweight="bold",
        y=0.985,
    )

    state_df = clean_df[
        (clean_df["disease_category"] == disease_category) & (clean_df["state"] == state)
    ]
    state_summary = summary_df[
        (summary_df["disease_category"] == disease_category) & (summary_df["state"] == state)
    ]

    subplot_positions = {
        "55-64 years": (0, 0),
        "65-74 years": (0, 1),
        "75-84 years": (1, 0),
        "85+ years": (1, 1),
    }

    legend_handles = {}
    for age_group in AGE_GROUPS:
        row, col = subplot_positions[age_group]
        ax = axes[row][col]
        panel_df = state_df[state_df["age_group"] == age_group]
        panel_summary = state_summary[state_summary["age_group"] == age_group]
        y_range = ranges.get((disease_category, age_group))

        ax.set_title(age_group, fontsize=19, fontweight="bold", pad=12)
        ax.grid(True, color="#dddddd", linewidth=0.8, alpha=0.7)
        ax.set_xlim(1998.5, 2025.5)
        ax.set_xticks([1999, 2005, 2010, 2015, 2020, 2025])
        ax.tick_params(axis="both", labelsize=12)
        ax.set_xlabel("Year", fontsize=13)
        ax.set_ylabel("Avg daily crude rate per 100,000", fontsize=13)
        if y_range is not None:
            ax.set_ylim(*y_range)

        has_any_series = False
        for gender in GENDER_ORDER:
            subset = panel_df[panel_df["gender"] == gender].sort_values("year")
            reg = panel_summary[panel_summary["gender"] == gender]
            if subset.empty or reg.empty:
                continue

            has_any_series = True
            reg_row = reg.iloc[0]
            years = subset["year"].to_numpy(dtype=int)
            observed = subset["average_daily_crude_rate"].to_numpy(dtype=float)
            fit = fitted_values(reg_row, years)
            color = COLORS[gender]

            observed_line = ax.plot(
                years,
                observed,
                color=color,
                linewidth=1.4,
                linestyle=":",
                marker="o",
                markersize=3.5,
                alpha=0.35,
                label=f"{gender} observed",
            )[0]
            fitted_line = ax.plot(
                years,
                fit,
                color=color,
                linewidth=3.0,
                alpha=0.95,
                label=f"{gender} fitted",
            )[0]
            legend_handles.setdefault(f"{gender} observed", observed_line)
            legend_handles.setdefault(f"{gender} fitted", fitted_line)

            breakpoint_year = int(reg_row["changepoint_year"])
            ax.axvline(
                breakpoint_year,
                color=color,
                linewidth=1.8,
                linestyle="--",
                alpha=0.85,
            )
            ha = "right" if breakpoint_year >= 2021 else "left"
            x_offset = -0.25 if ha == "right" else 0.25
            ax.text(
                breakpoint_year + x_offset,
                BREAKPOINT_LABEL_Y[gender],
                f"{gender} BP: {breakpoint_year}",
                transform=ax.get_xaxis_transform(),
                ha=ha,
                va="center",
                fontsize=11.5,
                color=color,
                bbox={
                    "boxstyle": "round,pad=0.22",
                    "facecolor": "white",
                    "edgecolor": color,
                    "alpha": 0.88,
                    "linewidth": 0.9,
                },
            )

        if not has_any_series:
            ax.text(
                0.5,
                0.5,
                "Insufficient usable years",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=15,
                color="#666666",
            )

    ordered_labels = [
        f"{gender} {kind}"
        for gender in GENDER_ORDER
        for kind in ["observed", "fitted"]
        if f"{gender} {kind}" in legend_handles
    ]
    fig.legend(
        [legend_handles[label] for label in ordered_labels],
        ordered_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=6,
        fontsize=13,
        frameon=False,
    )
    fig.text(
        0.5,
        0.018,
        "Dashed vertical lines mark gender-specific estimated breakpoints; dotted lines are observed rates and solid lines are segmented fitted rates.",
        ha="center",
        fontsize=13,
        color="#333333",
    )
    fig.tight_layout(rect=[0.02, 0.045, 0.98, 0.91])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_presentation_state_dashboards(
    clean_df: pd.DataFrame, summary_df: pd.DataFrame
) -> list[dict[str, str]]:
    output_root = PLOTS_DIR / "presentation_by_state"
    output_root.mkdir(parents=True, exist_ok=True)
    ranges = y_axis_ranges(clean_df)
    dashboard_links = []

    subplot_positions = {
        "55-64 years": (1, 1),
        "65-74 years": (1, 2),
        "75-84 years": (2, 1),
        "85+ years": (2, 2),
    }

    for disease_category in sorted(clean_df["disease_category"].unique()):
        disease_df = clean_df[clean_df["disease_category"] == disease_category]
        disease_summary = summary_df[summary_df["disease_category"] == disease_category]
        output_dir = output_root / disease_category
        output_dir.mkdir(parents=True, exist_ok=True)

        for state in sorted(disease_df["state"].dropna().unique()):
            state_df = disease_df[disease_df["state"] == state]
            state_summary = disease_summary[disease_summary["state"] == state]
            if state_df.empty:
                continue

            fig = make_subplots(
                rows=2,
                cols=2,
                subplot_titles=AGE_GROUPS,
                horizontal_spacing=0.08,
                vertical_spacing=0.14,
            )

            for age_group in AGE_GROUPS:
                row, col = subplot_positions[age_group]
                panel_df = state_df[state_df["age_group"] == age_group]
                panel_summary = state_summary[state_summary["age_group"] == age_group]
                y_range = ranges.get((disease_category, age_group))

                for gender in GENDER_ORDER:
                    subset = panel_df[panel_df["gender"] == gender].sort_values("year")
                    reg = panel_summary[panel_summary["gender"] == gender]
                    if subset.empty or reg.empty:
                        continue

                    reg_row = reg.iloc[0]
                    years = subset["year"].to_numpy(dtype=int)
                    fit = fitted_values(reg_row, years)
                    color = COLORS[gender]
                    showlegend = age_group == AGE_GROUPS[0]

                    fig.add_trace(
                        go.Scatter(
                            x=years,
                            y=subset["average_daily_crude_rate"],
                            mode="lines+markers",
                            name=f"{gender} observed",
                            legendgroup=f"{gender}_observed",
                            showlegend=showlegend,
                            line={"color": color, "width": 1.8, "dash": "dot"},
                            marker={"size": 5},
                            opacity=0.35,
                        ),
                        row=row,
                        col=col,
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=years,
                            y=fit,
                            mode="lines",
                            name=f"{gender} fitted",
                            legendgroup=f"{gender}_fitted",
                            showlegend=showlegend,
                            line={"color": color, "width": 4},
                        ),
                        row=row,
                        col=col,
                    )

                    breakpoint_year = int(reg_row["changepoint_year"])
                    fig.add_vline(
                        x=breakpoint_year,
                        line={"color": color, "width": 2, "dash": "dash"},
                        row=row,
                        col=col,
                    )
                    add_breakpoint_annotation(
                        fig, row, col, gender, breakpoint_year, y_range
                    )

                if y_range is not None:
                    fig.update_yaxes(range=y_range, row=row, col=col)
                fig.update_xaxes(
                    title_text="Year",
                    tickmode="array",
                    tickvals=[1999, 2005, 2010, 2015, 2020, 2025],
                    tickfont={"size": 13},
                    row=row,
                    col=col,
                )
                fig.update_yaxes(
                    title_text="Avg daily crude rate per 100,000",
                    tickfont={"size": 13},
                    row=row,
                    col=col,
                )

            disease_label = disease_category.replace("_", " ").title()
            title = f"{state}: {disease_label} Mortality by Age and Gender"
            fig.update_layout(
                title={"text": title, "x": 0.5, "font": {"size": 30}},
                template="plotly_white",
                width=1800,
                height=1250,
                font={"size": 16},
                legend={
                    "orientation": "h",
                    "yanchor": "bottom",
                    "y": 1.03,
                    "xanchor": "center",
                    "x": 0.5,
                    "font": {"size": 16},
                },
                margin={"l": 95, "r": 45, "t": 150, "b": 80},
            )
            for annotation in fig.layout.annotations:
                if annotation.text in AGE_GROUPS:
                    annotation.font = {"size": 22}

            filename_base = f"{slugify(state)}_age_dashboard"
            html_path = output_dir / f"{filename_base}.html"
            png_path = output_dir / f"{filename_base}.png"
            fig.write_html(html_path, include_plotlyjs="cdn")
            write_static_dashboard_png(
                clean_df, summary_df, disease_category, state, png_path, ranges
            )
            png_relative_path = f"presentation_by_state/{disease_category}/{png_path.name}"

            dashboard_links.append(
                {
                    "disease_category": disease_category,
                    "state": state,
                    "html_path": f"presentation_by_state/{disease_category}/{html_path.name}",
                    "png_path": png_relative_path,
                }
            )

    return dashboard_links


def write_presentation_index(dashboard_links: list[dict[str, str]]) -> None:
    output_root = PLOTS_DIR / "presentation_by_state"
    links_df = pd.DataFrame(dashboard_links)
    html = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        "<meta charset='utf-8'>",
        "<title>Presentation Dashboards by State</title>",
        "<style>",
        "body { font-family: Arial, sans-serif; margin: 36px; color: #222; }",
        "h1 { margin-bottom: 8px; }",
        "h2 { margin-top: 34px; border-bottom: 1px solid #ddd; padding-bottom: 6px; }",
        ".state-list { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 10px; }",
        ".state-link { border: 1px solid #ddd; border-radius: 6px; padding: 10px; line-height: 1.6; }",
        "a { color: #1769aa; text-decoration: none; }",
        "a:hover { text-decoration: underline; }",
        "</style>",
        "</head>",
        "<body>",
        "<h1>Presentation Dashboards by State</h1>",
        "<p>Each dashboard shows the four age groups for one disease category and state.</p>",
    ]

    for disease_category in sorted(links_df["disease_category"].unique()):
        disease_df = links_df[links_df["disease_category"] == disease_category]
        html.append(f"<h2>{disease_category.replace('_', ' ').title()}</h2>")
        html.append("<div class='state-list'>")
        for _, row in disease_df.sort_values("state").iterrows():
            html.append("<div class='state-link'>")
            html.append(f"<strong>{row['state']}</strong><br>")
            html.append(f"<a href='../{row['html_path']}'>HTML dashboard</a>")
            if row["png_path"]:
                html.append(f" | <a href='../{row['png_path']}'>PNG</a>")
            html.append("</div>")
        html.append("</div>")

    html.extend(["</body>", "</html>"])
    (output_root / "index.html").write_text("\n".join(html), encoding="utf-8")


def write_plot_index(plot_links: list[dict[str, str]]) -> None:
    html = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        "<meta charset='utf-8'>",
        "<title>Ischemic vs Heart Failure Segmented Regression Plots</title>",
        "<style>",
        "body { font-family: Arial, sans-serif; margin: 36px; color: #222; }",
        "h1 { margin-bottom: 8px; }",
        "h2 { margin-top: 34px; border-bottom: 1px solid #ddd; padding-bottom: 6px; }",
        "h3 { margin-top: 22px; }",
        ".state-list { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 10px; }",
        ".state-link { border: 1px solid #ddd; border-radius: 6px; padding: 10px; }",
        "a { color: #1769aa; text-decoration: none; }",
        "a:hover { text-decoration: underline; }",
        "</style>",
        "</head>",
        "<body>",
        "<h1>Ischemic vs Heart Failure Segmented Regression</h1>",
        "<p>Navigate by disease category, age group, and state.</p>",
    ]

    links_df = pd.DataFrame(plot_links)
    for disease_category in sorted(links_df["disease_category"].unique()):
        html.append(f"<h2>{disease_category.replace('_', ' ').title()}</h2>")
        disease_df = links_df[links_df["disease_category"] == disease_category]
        for age_group in AGE_GROUPS:
            age_df = disease_df[disease_df["age_group"] == age_group]
            if age_df.empty:
                continue
            html.append(f"<h3>{age_group}</h3>")
            html.append("<div class='state-list'>")
            for _, row in age_df.sort_values("state").iterrows():
                html.append(
                    f"<div class='state-link'><a href='{row['path']}'>{row['state']}</a></div>"
                )
            html.append("</div>")

    html.extend(["</body>", "</html>"])
    (PLOTS_DIR / "index.html").write_text("\n".join(html), encoding="utf-8")


def plot_breakpoint_distributions(summary_df: pd.DataFrame) -> None:
    output_root = PLOTS_DIR / "breakpoint_distributions"
    output_root.mkdir(parents=True, exist_ok=True)

    for (disease_category, age_group, gender), subset in summary_df.groupby(
        ["disease_category", "age_group", "gender"]
    ):
        fig = go.Figure(
            data=[
                go.Histogram(
                    x=subset["changepoint_year"],
                    xbins={
                        "start": int(subset["changepoint_year"].min()) - 0.5,
                        "end": int(subset["changepoint_year"].max()) + 0.5,
                        "size": 1,
                    },
                    marker={"color": COLORS.get(gender, "#555555")},
                )
            ]
        )
        fig.update_layout(
            title=(
                f"Breakpoint Distribution: {disease_category.replace('_', ' ').title()}, "
                f"{age_group}, {gender}"
            ),
            xaxis_title="Changepoint year",
            yaxis_title="Number of states",
            template="plotly_white",
        )
        output_dir = output_root / disease_category / AGE_SLUGS[age_group]
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.write_html(output_dir / f"{slugify(gender)}.html", include_plotlyjs="cdn")


def slope_direction(significance_label: str) -> str:
    if significance_label == "significant_increase":
        return "rising"
    if significance_label == "significant_decrease":
        return "falling"
    return "not_significant"


def write_divergence_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    male_female = summary_df[summary_df["gender"].isin(["Male", "Female"])].copy()
    male_female["direction"] = male_female["significance_label"].map(slope_direction)
    rows = []
    for keys, group in male_female.groupby(["disease_category", "age_group", "state"]):
        pivot = group.set_index("gender")
        if not {"Male", "Female"}.issubset(pivot.index):
            continue
        disease_category, age_group, state = keys
        male = pivot.loc["Male"]
        female = pivot.loc["Female"]
        rows.append(
            {
                "disease_category": disease_category,
                "age_group": age_group,
                "state": state,
                "male_changepoint_year": male["changepoint_year"],
                "female_changepoint_year": female["changepoint_year"],
                "male_slope_after": male["slope_after"],
                "female_slope_after": female["slope_after"],
                "male_slope_after_p_value": male["slope_after_p_value"],
                "female_slope_after_p_value": female["slope_after_p_value"],
                "male_direction": male["direction"],
                "female_direction": female["direction"],
                "divergence_label": (
                    "same_direction"
                    if male["direction"] == female["direction"]
                    else "divergent_direction"
                ),
            }
        )
    divergence_df = pd.DataFrame(rows).sort_values(["disease_category", "age_group", "state"])
    divergence_df.to_csv(DIVERGENCE_OUTPUT, index=False)
    return divergence_df


def main() -> None:
    clean_df = load_clean_dataset()
    clean_df.to_csv(CLEAN_OUTPUT, index=False)

    summary_df, validation_df = run_segmented_regressions(clean_df)
    validation_df.to_csv(VALIDATION_OUTPUT, index=False)

    model_comparison_df = run_model_comparison(clean_df)
    model_comparison_df.to_csv(MODEL_COMPARISON_OUTPUT, index=False)
    write_model_comparison_qa(model_comparison_df)

    anova_df = run_anova_model_comparison(clean_df, model_comparison_df)
    anova_df.to_csv(ANOVA_COMPARISON_OUTPUT, index=False)
    write_anova_qa(anova_df)

    summary_df = add_anova_results_to_summary(summary_df, anova_df)
    summary_df.to_csv(SUMMARY_OUTPUT, index=False)
    complete_summary_df = write_complete_changepoint_summary(clean_df, summary_df)
    write_b2_significance_outputs(summary_df)

    slope_after_df = run_reversed_hinge_slope_after_significance(clean_df, summary_df)
    write_slope_after_significance_outputs(slope_after_df)
    master_df = write_tableau_master(
        complete_summary_df, slope_after_df, model_comparison_df
    )

    plot_links = plot_segmented_states(clean_df, summary_df)
    write_plot_index(plot_links)
    dashboard_links = plot_presentation_state_dashboards(clean_df, summary_df)
    write_presentation_index(dashboard_links)
    plot_breakpoint_distributions(summary_df)
    write_divergence_table(summary_df)

    max_diffs = validation_df[
        [
            "intercept_abs_diff",
            "slope_before_abs_diff",
            "change_in_slope_abs_diff",
            "slope_after_abs_diff",
            "sse_abs_diff",
        ]
    ].max()

    print("Pipeline complete")
    print(f"Clean rows: {len(clean_df):,}")
    print(f"Fitted model rows: {len(summary_df):,}")
    print(f"Complete model grid rows: {len(complete_summary_df):,}")
    print(f"Tableau master rows: {len(master_df):,}")
    print(f"Sparse fitted segments flagged: {int(summary_df['sparse_segment_flag'].sum()):,}")
    print(f"State plot files: {len(plot_links):,}")
    print(f"Presentation dashboards: {len(dashboard_links):,}")
    print(f"Main outputs:\n  {CLEAN_OUTPUT}\n  {SUMMARY_OUTPUT}\n  {TABLEAU_MASTER_OUTPUT}")
    print("Statsmodels vs. numpy least-squares max absolute differences:")
    for name, value in max_diffs.items():
        print(f"  {name}: {value:.12g}")


if __name__ == "__main__":
    main()
