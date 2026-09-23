"""Input adapter between processed scenario tables and the optimizer notebook."""

from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from optimizer.configuration import (
    CsvTableStore,
    load_active_runtime,
    route_bess,
    route_generation,
    route_price,
    source_stem,
)


ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "1_Dataset" / "2_Processed_data"


def _dated(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "Date" not in frame:
        raise KeyError(f"{path.name} must contain a Date column.")
    frame["Date"] = pd.to_datetime(frame["Date"])
    return frame


def _align(frame: pd.DataFrame, dates: pd.Series, path: Path) -> pd.DataFrame:
    aligned = pd.DataFrame({"Date": pd.to_datetime(dates)}).merge(
        frame, on="Date", how="left", validate="one_to_one"
    )
    if aligned.drop(columns="Date").isna().any().any():
        raise ValueError(f"{path.name} does not align with Dataset/dataset.csv.")
    return aligned


def _binary_instruction(predicted: pd.Series, duration_hours: float, granularity_minutes: int) -> pd.Series:
    intervals = max(1, int(round(duration_hours * 60 / granularity_minutes)))
    per_day = int(24 * 60 / granularity_minutes)
    work = pd.DataFrame({"price": pd.to_numeric(predicted, errors="raise")})
    work["day"] = np.arange(len(work)) // per_day
    work["rank"] = work.groupby("day")["price"].rank(method="first")
    counts = work.groupby("day")["price"].transform("size")
    instruction = pd.Series(0, index=work.index, dtype=int)
    instruction.loc[work["rank"] <= intervals] = 1
    instruction.loc[work["rank"] > counts - intervals] = 2
    return instruction


def prepare_optimizer_inputs(processed_dir: Path | str = PROCESSED) -> dict:
    processed = Path(processed_dir)
    settings = load_active_runtime()
    dataset = _dated(processed / "Dataset" / "dataset.csv")
    runtime = CsvTableStore().runtime()
    configuration, asset = runtime["configuration"], runtime["asset"]
    price_code = route_price(configuration)
    actual_price_code = "A1M" if price_code == "A2M" else "A1" if price_code == "A2" else price_code
    generation_code, bess_code = route_generation(configuration), route_bess(configuration)
    actual_path = processed / f"{source_stem(actual_price_code, asset)}.csv"
    predicted_path = processed / f"{source_stem(price_code, asset)}.csv"
    generation_path = processed / f"{source_stem(generation_code, asset)}.csv"
    bess_path = processed / f"{source_stem(bess_code, asset)}.csv"
    price_actual = _align(_dated(actual_path), dataset["Date"], actual_path)
    price_predicted = _align(_dated(predicted_path), dataset["Date"], predicted_path)
    generation = _align(_dated(generation_path), dataset["Date"], generation_path)
    bess_sizes = pd.read_csv(bess_path)
    price_columns = [column for column in price_predicted if column != "Date"]
    generation_columns = [column for column in generation if column != "Date"]
    if set(price_columns) != set(column for column in price_actual if column != "Date"):
        raise ValueError("Actual and predicted price scenario columns must match.")
    if not price_columns or not generation_columns or bess_sizes.empty:
        raise ValueError("At least one price, generation, and BESS scenario is required.")
    return {
        "dataset": dataset,
        "price_actual": price_actual,
        "price_predicted": price_predicted,
        "generation": generation,
        "bess_sizes": bess_sizes,
        "items": list(product(price_columns, generation_columns, range(len(bess_sizes)))),
        "settings": settings,
    }


def select_optimization_case(inputs: dict, item=None):
    price_column, generation_column, bess_index = item or inputs["items"][0]
    dataset = inputs["dataset"].copy()
    dataset["Actual_price"] = inputs["price_actual"][price_column].to_numpy()
    dataset["Predicted_price"] = inputs["price_predicted"][price_column].to_numpy()
    dataset["Generation"] = inputs["generation"][generation_column].to_numpy()
    bess = inputs["bess_sizes"].iloc[int(bess_index)]
    settings = inputs["settings"]
    if settings["prediction_signal_transform"] == 2:
        dataset["BESS_instruction"] = _binary_instruction(
            dataset["Predicted_price"],
            float(bess["BESS_Duration_h"]),
            int(settings["operation_granularity_in_minutes"]),
        )
    else:
        dataset["BESS_instruction"] = 0
    dataset["Generation_exportable"] = np.clip(
        dataset["Generation"] - settings["project_poi_export_limit_mwh_per_interval"],
        a_min=0,
        a_max=None,
    )
    label = f"price={price_column}|generation={generation_column}|bess={bess_index}"
    return (
        dataset,
        float(bess["BESS_MWh"]),
        float(bess["BESS_MWh_per_interval"]),
        label,
        bess,
    )
