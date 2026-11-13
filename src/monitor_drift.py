import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

REFERENCE_FILE = PROJECT_ROOT / "data" / "processed" / "train.csv"
CURRENT_FILE = PROJECT_ROOT / "data" / "processed" / "test.csv"

REPORT_FILE = PROJECT_ROOT / "reports" / "drift_report.json"
LOG_FILE = PROJECT_ROOT / "reports" / "drift_monitoring.log"

MODERATE_DRIFT = 0.10
SIGNIFICANT_DRIFT = 0.20

TARGET_COLUMN = "late_delivery"


# --------------------------------------------------
# LOGGING
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


# --------------------------------------------------
# POPULATION STABILITY INDEX
# --------------------------------------------------

def calculate_psi(reference, current):

    reference = pd.Series(reference).dropna()
    current = pd.Series(current).dropna()

    if len(reference) == 0 or len(current) == 0:
        return 0.0

    if pd.api.types.is_numeric_dtype(reference):

        quantiles = np.unique(
            np.quantile(
                reference,
                np.linspace(0, 1, 11)
            )
        )

        if len(quantiles) < 2:
            return 0.0

        bins = np.concatenate((
            [-np.inf],
            quantiles[1:-1],
            [np.inf]
        ))

        ref_counts, _ = np.histogram(reference, bins=bins)
        cur_counts, _ = np.histogram(current, bins=bins)

    else:

        categories = sorted(
            set(reference.astype(str)) |
            set(current.astype(str))
        )

        ref_counts = (
            reference.astype(str)
            .value_counts()
            .reindex(categories, fill_value=0)
            .values
        )

        cur_counts = (
            current.astype(str)
            .value_counts()
            .reindex(categories, fill_value=0)
            .values
        )

    epsilon = 0.0001

    ref_percent = (
        ref_counts + epsilon
    ) / (ref_counts.sum() + epsilon * len(ref_counts))

    cur_percent = (
        cur_counts + epsilon
    ) / (cur_counts.sum() + epsilon * len(cur_counts))

    psi = np.sum(
        (cur_percent - ref_percent) *
        np.log(cur_percent / ref_percent)
    )

    return float(psi)


# --------------------------------------------------
# DRIFT MONITORING
# --------------------------------------------------

def monitor_drift(reference_file, current_file):

    logger.info("STARTING DATA DRIFT MONITORING")

    reference = pd.read_csv(reference_file)
    current = pd.read_csv(current_file)

    # Target is excluded: only model input features are monitored.
    reference = reference.drop(
        columns=[TARGET_COLUMN],
        errors="ignore"
    )

    current = current.drop(
        columns=[TARGET_COLUMN],
        errors="ignore"
    )

    common_features = [
        column for column in reference.columns
        if column in current.columns
    ]

    if not common_features:
        raise ValueError("No common features found.")

    logger.info("Reference records: %s", len(reference))
    logger.info("Current records: %s", len(current))

    results = {}

    for feature in common_features:

        psi = calculate_psi(
            reference[feature],
            current[feature]
        )

        if psi >= SIGNIFICANT_DRIFT:
            status = "SIGNIFICANT DRIFT"

            logger.warning(
                "%s: Significant drift detected (PSI=%.4f)",
                feature,
                psi
            )

        elif psi >= MODERATE_DRIFT:
            status = "MODERATE DRIFT"

            logger.warning(
                "%s: Moderate drift detected (PSI=%.4f)",
                feature,
                psi
            )

        else:
            status = "NO SIGNIFICANT DRIFT"

            logger.info(
                "%s: No significant drift (PSI=%.4f)",
                feature,
                psi
            )

        results[feature] = {
            "psi": round(psi, 6),
            "status": status
        }

    drift_count = sum(
        result["status"] != "NO SIGNIFICANT DRIFT"
        for result in results.values()
    )

    report = {
        "reference_records": len(reference),
        "current_records": len(current),
        "features_monitored": len(common_features),
        "features_with_drift": drift_count,
        "drift_results": results
    }

    REPORT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(REPORT_FILE, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    logger.info("DRIFT MONITORING COMPLETED")
    logger.info("Features monitored: %s", len(common_features))
    logger.info("Features with drift: %s", drift_count)
    logger.info("Report saved to: %s", REPORT_FILE)


# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Monitor shipment data drift."
    )

    parser.add_argument(
        "--reference",
        default=str(REFERENCE_FILE)
    )

    parser.add_argument(
        "--current",
        default=str(CURRENT_FILE)
    )

    args = parser.parse_args()

    monitor_drift(
        args.reference,
        args.current
    )