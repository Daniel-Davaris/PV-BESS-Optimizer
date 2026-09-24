"""CSV-backed configuration, dynamic data loading, and notebook UI.

The two CSV tables in ``configuration_tables`` are the source of truth. This
module deliberately does not import ``optimizer.variables`` so that the latter
can remain a small compatibility adapter for the optimisation notebook.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "1_Dataset"
RAW_DATA_DIR = DATASET_DIR / "1_Raw_data"
PROCESSED_DATA_DIR = DATASET_DIR / "2_Processed_data"
TABLE_DIR = ROOT / "configuration_tables"

STATES = ("NSW", "VIC", "QLD", "SA", "TAS")
PRICE_TIMEFRAMES = ("Historical back-test", "Future projection")
FORESIGHT_TYPES = ("Perfect foresight", "Realistic foresight")
GENERATION_PROFILES = ("Standard year", "Observed values")
PREDICTION_TRANSFORMS = ("Normal predicted value", "Transform to binary signal", "Both")
LGC_ELIGIBILITY_BASES = (
    "Generation", "Renewable dispatched", "Net export", "TLF-adjusted net export"
)

# Current naming convention. Internal B1S/B1SM keys distinguish the B1
# standard-year variants from observed historical generation.
SOURCE_TEMPLATES = {
    "A1": "A1_historical_price",
    "A1M": "A1M_historical_price_resampled_{state}",
    "A2": "A2_future_price_ST_predicted",
    "A2M": "A2M_future_price_ST_predicted_resampled_{state}",
    "A3": "A3_future_price_LT_predicted",
    "A3M": "A3M_future_price_LT_predicted_resampled_{state}",
    "B1": "B1_historical_generation",
    "B1M": "B1M_historical_generation_resampled_{asset}",
    "B1S": "B1_historical_generation_standard_year",
    "B1SM": "B1M_historical_generation_standard_year_resampled_{asset}",
    "B2": "B2_future_generation_ST_predicted",
    "B2M": "B2M_future_generation_ST_predicted_resampled_{asset}",
    "B3": "B3_future_generation_LT_predicted",
    "B3M": "B3M_future_generation_LT_predicted_resampled_{asset}",
    "C1": "C1_LGC",
    "D1": "D1_network_time_of_use_tarrif_mapping",
    "D2": "D2_network_tarrif_values",
    "E1": "E1_physical_constraints_mapping",
    "F1": "F1_BESS_size",
    "F1M": "F1M_BESS_size_resampled_{asset}",
}

ASSET_COLUMNS = [
    "asset_name", "state", "tariff_code", "demand_charge_off_peak",
    "demand_charge_shoulder", "demand_charge_peak", "export_charge_sun_soaker",
    "project_poi_export_limit_MW", "project_poi_import_limit_MW",
    "project_poi_export_limit_MWh_per_interval",
    "project_poi_import_limit_MWh_per_interval", "grid_import_penalty",
    "optimize_with_DUOS",
    "TLF_PV_injection", "TLF_BESS_injection", "TLF_BESS_withdrawal",
    "price_escalation_factor", "LGC_eligibility_basis",
    "LGC_price_override_per_MWh",
    "BESS_technology", "BESS_coupling_type", "BESS_duration_hours",
    "BESS_power_MW", "BESS_usable_fraction", "BESS_energy_override_MWh",
    "BESS_Start_SoC",
    "BESS_cycles_per_day", "BESS_lifetime_years",
]

CONFIGURATION_COLUMNS = [
    "configuration_name", "active", "asset", "execution_mode",
    "display_window_scheduler_visual", "optimization_start_date",
    "optimization_end_date", "optimization_granularity_in_minutes",
    "dataset_date_range", "dataset_year_month_mapping", "dataset_index_range",
    "optimization_horizon_hours", "optimization_avoid_edge_effect_hours",
    "number_of_intervals_per_window", "price_timeframe", "price_foresight",
    "price_run_synthetic_distribution", "price_series_start", "price_series_stop",
    "trade_signal_penalty", "prediction_signal_transform", "curtailment_penalty",
    "generation_timeframe", "generation_foresight", "generation_profile",
    "generation_run_synthetic_distribution", "generation_series_start",
    "generation_series_stop", "bess_run_synthetic_distribution",
    "bess_series_start", "bess_series_stop", "price_source", "generation_source",
    "bess_source",
]


def _bool(value) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _token(value: object, lower: bool = False) -> str:
    token = re.sub(r"\s+", "_", str(value).strip())
    return token.lower() if lower else token


def _date(value: object) -> pd.Timestamp:
    return pd.to_datetime(value, dayfirst=True)


def route_price(configuration: dict) -> str:
    multi = _bool(configuration["price_run_synthetic_distribution"])
    if configuration["price_timeframe"] == "Future projection":
        return "A3M" if multi else "A3"
    if configuration["price_foresight"] == "Realistic foresight":
        return "A2M" if multi else "A2"
    return "A1M" if multi else "A1"


def route_generation(configuration: dict) -> str:
    multi = _bool(configuration["generation_run_synthetic_distribution"])
    if configuration["generation_timeframe"] == "Future projection":
        return "B3M" if multi else "B3"
    if configuration["generation_foresight"] == "Realistic foresight":
        return "B2M" if multi else "B2"
    if configuration["generation_profile"] == "Standard year":
        return "B1SM" if multi else "B1S"
    return "B1M" if multi else "B1"


def route_bess(configuration: dict) -> str:
    return "F1M" if _bool(configuration["bess_run_synthetic_distribution"]) else "F1"


def source_stem(code: str, asset: dict) -> str:
    return SOURCE_TEMPLATES[code].format(
        state=_token(asset["state"], lower=True), asset=_token(asset["asset_name"])
    )


def source_candidates(code: str, asset: dict) -> list[Path]:
    stem = source_stem(code, asset)
    return [RAW_DATA_DIR / f"{stem}.csv"]


def resolve_lookup_source(code: str) -> Path:
    dummy = {"state": "NSW", "asset_name": "asset"}
    for path in source_candidates(code, dummy):
        if path.is_file():
            return path
    raise FileNotFoundError(f"Missing lookup source {SOURCE_TEMPLATES[code]}.csv in {RAW_DATA_DIR}.")


class CsvTableStore:
    """CRUD and validation for the asset/configuration CSV tables."""

    def __init__(self, table_dir: Path | str = TABLE_DIR):
        self.table_dir = Path(table_dir)
        self.asset_path = self.table_dir / "assets.csv"
        self.configuration_path = self.table_dir / "configurations.csv"
        self.table_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_tables()

    def _ensure_tables(self):
        if not self.asset_path.exists():
            pd.DataFrame(columns=ASSET_COLUMNS).to_csv(self.asset_path, index=False)
        if not self.configuration_path.exists():
            pd.DataFrame(columns=CONFIGURATION_COLUMNS).to_csv(self.configuration_path, index=False)

    @staticmethod
    def _read(path: Path, columns: list[str]) -> pd.DataFrame:
        frame = pd.read_csv(path, keep_default_na=False)
        for column in columns:
            if column not in frame:
                frame[column] = ""
        return frame[columns]

    def assets(self) -> pd.DataFrame:
        return self._read(self.asset_path, ASSET_COLUMNS)

    def configurations(self) -> pd.DataFrame:
        return self._read(self.configuration_path, CONFIGURATION_COLUMNS)

    def tariff_codes(self) -> list[str]:
        return sorted(pd.read_csv(resolve_lookup_source("D2"))["Tariff code"].dropna().astype(str).unique())

    def bess_technologies(self) -> list[str]:
        return sorted(pd.read_csv(resolve_lookup_source("E1"))["Project"].dropna().astype(str).unique())

    def _tariff_autos(self, tariff_code: str) -> dict:
        values = pd.read_csv(resolve_lookup_source("D2"))
        values = values.loc[values["Tariff code"].astype(str) == str(tariff_code)].copy()
        if values.empty:
            raise ValueError(f"Tariff code {tariff_code!r} is not present in D2.")

        def value(tou: str, column: str) -> float:
            row = values.loc[values["ToU"] == tou, column]
            return float(row.iloc[-1]) * 1000 if not row.empty else 0.0

        return {
            "demand_charge_off_peak": value("Off-peak", "Demand charge"),
            "demand_charge_shoulder": value("Shoulder", "Demand charge"),
            "demand_charge_peak": value("Peak", "Demand charge"),
            "export_charge_sun_soaker": value("Sun Soaker", "Export Charge"),
        }

    def save_asset(self, values: dict, interval_minutes: int = 60) -> dict:
        record = dict(values)
        name = str(record.get("asset_name", "")).strip()
        if not name:
            raise ValueError("asset_name is required.")
        record["asset_name"] = name
        if record.get("state") not in STATES:
            raise ValueError(f"state must be one of {STATES}.")
        if record.get("BESS_coupling_type") not in {"AC", "DC"}:
            raise ValueError("BESS_coupling_type must be AC or DC.")
        if record.get("LGC_eligibility_basis") not in LGC_ELIGIBILITY_BASES:
            raise ValueError(f"LGC_eligibility_basis must be one of {LGC_ELIGIBILITY_BASES}.")
        record["optimize_with_DUOS"] = _bool(record.get("optimize_with_DUOS", True))
        for field in (
            "project_poi_export_limit_MW", "project_poi_import_limit_MW",
            "grid_import_penalty", "BESS_duration_hours", "BESS_power_MW",
            "BESS_cycles_per_day", "BESS_lifetime_years", "TLF_PV_injection",
            "TLF_BESS_injection", "TLF_BESS_withdrawal", "price_escalation_factor",
            "LGC_price_override_per_MWh", "BESS_energy_override_MWh",
        ):
            record[field] = float(record[field])
            if record[field] < 0:
                raise ValueError(f"{field} cannot be negative.")
        if record["price_escalation_factor"] <= 0:
            raise ValueError("price_escalation_factor must be greater than zero.")
        record["BESS_Start_SoC"] = float(record["BESS_Start_SoC"])
        if not 0 <= record["BESS_Start_SoC"] <= 1:
            raise ValueError("BESS_Start_SoC must be between 0 and 1.")
        record["BESS_usable_fraction"] = float(record["BESS_usable_fraction"])
        if not 0 < record["BESS_usable_fraction"] <= 1:
            raise ValueError("BESS_usable_fraction must be greater than 0 and no more than 1.")
        record.update(self._tariff_autos(str(record["tariff_code"])))
        interval_fraction = int(interval_minutes) / 60
        record["project_poi_export_limit_MWh_per_interval"] = record["project_poi_export_limit_MW"] * interval_fraction
        record["project_poi_import_limit_MWh_per_interval"] = record["project_poi_import_limit_MW"] * interval_fraction
        frame = self.assets()
        existing = frame["asset_name"].astype(str).str.strip() == name
        row = {column: record.get(column, "") for column in ASSET_COLUMNS}
        if existing.any():
            matching_indices = frame.index[existing]
            frame.loc[matching_indices[0], ASSET_COLUMNS] = [row[column] for column in ASSET_COLUMNS]
            frame = frame.drop(matching_indices[1:])
        else:
            frame = pd.concat([frame, pd.DataFrame([row])], ignore_index=True)
        frame.to_csv(self.asset_path, index=False)
        return row

    @staticmethod
    def _derived_configuration(record: dict) -> dict:
        start, end = _date(record["optimization_start_date"]), _date(record["optimization_end_date"])
        granularity = int(record["optimization_granularity_in_minutes"])
        if end <= start:
            raise ValueError("optimization_end_date must be after optimization_start_date.")
        if granularity <= 0 or 60 % granularity != 0:
            raise ValueError("optimization_granularity_in_minutes must be a positive divisor of 60.")
        horizon = float(record["optimization_horizon_hours"])
        edge = float(record["optimization_avoid_edge_effect_hours"])
        if horizon <= 0 or edge < 0:
            raise ValueError("The optimization horizon must be positive and edge-effect hours cannot be negative.")
        for prefix in ("price", "generation", "bess"):
            if int(record[f"{prefix}_series_stop"]) <= int(record[f"{prefix}_series_start"]):
                raise ValueError(f"{prefix}_series_stop must be greater than {prefix}_series_start.")
        months = pd.period_range(start.to_period("M"), end.to_period("M"), freq="M")
        interval_count = int((end - start).total_seconds() // (granularity * 60)) + 1
        record.update({
            "optimization_start_date": start.strftime("%d/%m/%Y %H:%M"),
            "optimization_end_date": end.strftime("%d/%m/%Y %H:%M"),
            "optimization_granularity_in_minutes": granularity,
            "dataset_date_range": f"{start.isoformat()}|{end.isoformat()}|{granularity}min",
            "dataset_year_month_mapping": ";".join(str(month) for month in months),
            "dataset_index_range": f"0:{interval_count - 1}",
            "number_of_intervals_per_window": int((horizon + edge) * 60 / granularity),
            "price_source": route_price(record),
            "generation_source": route_generation(record),
            "bess_source": route_bess(record),
        })
        return record

    def save_configuration(self, values: dict) -> dict:
        record = dict(values)
        name = str(record.get("configuration_name", "")).strip()
        if not name:
            raise ValueError("configuration_name is required.")
        record["configuration_name"] = name
        if str(record.get("asset", "")) not in set(self.assets()["asset_name"].astype(str)):
            raise ValueError("Select an asset that exists in assets.csv.")
        if record.get("price_timeframe") not in PRICE_TIMEFRAMES:
            raise ValueError("Invalid price_timeframe.")
        if record.get("generation_timeframe") not in PRICE_TIMEFRAMES:
            raise ValueError("Invalid generation_timeframe.")
        if record.get("prediction_signal_transform") not in PREDICTION_TRANSFORMS:
            raise ValueError("Invalid prediction_signal_transform.")
        for field in (
            "active", "display_window_scheduler_visual", "price_run_synthetic_distribution",
            "generation_run_synthetic_distribution", "bess_run_synthetic_distribution",
        ):
            record[field] = _bool(record.get(field, False))
        record = self._derived_configuration(record)
        frame = self.configurations()
        if record["active"]:
            frame["active"] = False
        existing = frame["configuration_name"].astype(str).str.strip() == name
        row = {column: record.get(column, "") for column in CONFIGURATION_COLUMNS}
        if existing.any():
            matching_indices = frame.index[existing]
            frame.loc[matching_indices[0], CONFIGURATION_COLUMNS] = [row[column] for column in CONFIGURATION_COLUMNS]
            frame = frame.drop(matching_indices[1:])
        else:
            frame = pd.concat([frame, pd.DataFrame([row])], ignore_index=True)
        if not frame["active"].map(_bool).any() and len(frame):
            frame.loc[frame["configuration_name"] == name, "active"] = True
            row["active"] = True
        frame.to_csv(self.configuration_path, index=False)
        return row

    def active_configuration(self) -> dict:
        frame = self.configurations()
        active = frame.loc[frame["active"].map(_bool)]
        if len(active) != 1:
            raise RuntimeError("configurations.csv must contain exactly one active configuration.")
        return active.iloc[0].to_dict()

    def runtime(self, configuration_name: str | None = None) -> dict:
        configurations = self.configurations()
        if configuration_name is None:
            configuration = self.active_configuration()
        else:
            match = configurations.loc[configurations["configuration_name"].astype(str) == str(configuration_name)]
            if match.empty:
                raise KeyError(f"Unknown configuration {configuration_name!r}.")
            configuration = match.iloc[0].to_dict()
        match = self.assets().loc[lambda value: value["asset_name"].astype(str) == str(configuration["asset"])]
        if match.empty:
            raise KeyError(f"Configuration refers to missing asset {configuration['asset']!r}.")
        return {"configuration": configuration, "asset": match.iloc[0].to_dict()}


@dataclass
class LoadedSources:
    dataset: pd.DataFrame
    price_actual: pd.DataFrame
    price_predicted: pd.DataFrame
    generation: pd.DataFrame
    bess_sizes: pd.DataFrame
    lgc: pd.DataFrame
    tariff: pd.DataFrame
    physical_constraints: pd.DataFrame


class DynamicDataLoader:
    """Load decision-map sources and build the optimizer handoff tables."""

    def __init__(self, configuration_name: str | None = None, store: CsvTableStore | None = None):
        self.store = store or CsvTableStore()
        runtime = self.store.runtime(configuration_name)
        self.configuration, self.asset = runtime["configuration"], runtime["asset"]
        # Decision fields are authoritative; source columns are cached display
        # values and are recomputed in case a CSV was edited outside the UI.
        self.configuration["price_source"] = route_price(self.configuration)
        self.configuration["generation_source"] = route_generation(self.configuration)
        self.configuration["bess_source"] = route_bess(self.configuration)
        self.start = _date(self.configuration["optimization_start_date"])
        self.end = _date(self.configuration["optimization_end_date"])
        self.granularity = int(self.configuration["optimization_granularity_in_minutes"])
        self.frequency = f"{self.granularity}min"
        self.target_dates = pd.date_range(self.start, self.end, freq=self.frequency)
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    def resolve_source(self, code: str) -> Path:
        for path in source_candidates(code, self.asset):
            if path.is_file():
                return path
        expected = " or ".join(path.name for path in source_candidates(code, self.asset))
        raise FileNotFoundError(f"No source for {code}. Expected {expected} in {RAW_DATA_DIR}.")

    def inventory(self) -> pd.DataFrame:
        selected = {self.configuration["price_source"], self.configuration["generation_source"], self.configuration["bess_source"], "C1", "D1", "D2", "E1"}
        rows = []
        for code in SOURCE_TEMPLATES:
            matches = [path for path in source_candidates(code, self.asset) if path.is_file()]
            rows.append({"Code": code, "Selected": code in selected, "Status": "available" if matches or code == "F1" else "missing", "File": matches[0].name if matches else source_candidates(code, self.asset)[0].name})
        return pd.DataFrame(rows)

    @staticmethod
    def _parse_dates(values: pd.Series, source: str) -> pd.Series:
        text = values.astype(str).str.strip()
        iso = text.str.match(r"^\d{4}-\d{1,2}-\d{1,2}")
        result = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
        result.loc[iso] = pd.to_datetime(text.loc[iso], format="mixed", yearfirst=True, errors="coerce")
        result.loc[~iso] = pd.to_datetime(text.loc[~iso], format="mixed", dayfirst=True, errors="coerce")
        if result.isna().any():
            raise ValueError(f"Unparseable dates in {source}: {values.loc[result.isna()].head(5).tolist()}")
        return result

    def _resample(self, frame: pd.DataFrame, columns: list[str], method: str, source: str, interval_ending: bool = False) -> pd.DataFrame:
        missing = [column for column in ["Date", *columns] if column not in frame]
        if missing:
            raise KeyError(f"{source} is missing columns: {missing}")
        work = frame[["Date", *columns]].copy()
        work["Date"] = self._parse_dates(work["Date"], source)
        work = work.sort_values("Date").drop_duplicates("Date", keep="last").set_index("Date")
        numeric = work[columns].apply(pd.to_numeric, errors="coerce")
        if method == "mean":
            resample_kwargs = {"closed": "right", "label": "left"} if interval_ending else {}
            result = numeric.resample(self.frequency, **resample_kwargs).mean()
        elif method == "ffill":
            result = numeric.resample(self.frequency).ffill()
        elif method == "interpolate":
            result = numeric.resample(self.frequency).mean().interpolate(method="time")
        else:
            raise ValueError(f"Unsupported resampling method {method!r}.")
        if method != "ffill":
            # Fill finer-granularity intervals while the source points on both
            # sides of the configured range are still available.
            result = result.interpolate(method="time", limit_area="inside")
        result = result.reindex(self.target_dates)
        result = result.ffill() if method == "ffill" else result.interpolate(method="time", limit_area="inside")
        if result.isna().any().any():
            bad = result.columns[result.isna().any()].tolist()
            raise ValueError(f"{source} does not cover the configured range for columns {bad}.")
        return result.rename_axis("Date").reset_index()

    def _resample_generation(self, frame: pd.DataFrame, columns: list[str], source: str) -> pd.DataFrame:
        """Convert average generation power into energy for each optimizer interval."""
        missing = [column for column in ["Date", *columns] if column not in frame]
        if missing:
            raise KeyError(f"{source} is missing columns: {missing}")
        work = frame[["Date", *columns]].copy()
        work["Date"] = self._parse_dates(work["Date"], source)
        work = work.sort_values("Date").drop_duplicates("Date", keep="last").set_index("Date")
        numeric = work[columns].apply(pd.to_numeric, errors="coerce")
        source_intervals = numeric.index.to_series().diff().dropna()
        if source_intervals.empty:
            raise ValueError(f"{source} needs at least two dated rows to determine its interval.")
        source_interval = source_intervals.median()
        target_interval = pd.Timedelta(minutes=self.granularity)
        if target_interval < source_interval:
            # Generation values represent average MW over their source
            # interval. Hold that power constant when splitting the interval.
            result = numeric.resample(self.frequency).ffill()
        else:
            result = numeric.resample(self.frequency).mean()
        result = result.reindex(self.target_dates)
        result = result.interpolate(method="time", limit_area="inside")
        if result.isna().any().any():
            bad = result.columns[result.isna().any()].tolist()
            raise ValueError(f"{source} does not cover the configured range for columns {bad}.")
        # The optimizer's flow variables are MWh per interval, not MW.
        result[columns] = result[columns] * (self.granularity / 60)
        return result.rename_axis("Date").reset_index()

    @staticmethod
    def _sample_columns(frame: pd.DataFrame, start: int, stop: int) -> list[str]:
        columns = [column for column in frame if re.fullmatch(r"Sample \d+", str(column))]
        columns.sort(key=lambda value: int(str(value).split()[-1]))
        selected = columns[int(start):int(stop)]
        if not selected:
            raise ValueError(f"Series range [{start}, {stop}) selects none of {len(columns)} sample columns.")
        return selected

    def _price_column(self, frame: pd.DataFrame) -> str:
        state = str(self.asset["state"]).upper()
        candidates = [column for column in frame if str(column).upper().startswith(state)]
        candidates += [f"{state.lower()}_price", state, "Price"]
        candidates = list(dict.fromkeys(column for column in candidates if column in frame))
        if not candidates:
            raise KeyError(f"No {state} price column is present.")
        dated = frame.copy()
        dated["Date"] = self._parse_dates(dated["Date"], "price source")
        in_range = dated.loc[dated["Date"].between(self.start, self.end)]
        coverage = {column: pd.to_numeric(in_range[column], errors="coerce").notna().sum() for column in candidates}
        return max(coverage, key=coverage.get)

    def load_price(self, code: str) -> pd.DataFrame:
        path = self.resolve_source(code)
        frame = pd.read_csv(path)
        if code.endswith("M"):
            columns = self._sample_columns(frame, int(self.configuration["price_series_start"]), int(self.configuration["price_series_stop"]))
            return self._resample(frame, columns, "mean", path.name)
        column = self._price_column(frame)
        return self._resample(frame, [column], "mean", path.name, interval_ending=(code == "A1")).rename(columns={column: "Scenario 1"})

    def _load_standard_year(self, code: str, path: Path, frame: pd.DataFrame) -> pd.DataFrame:
        if code.endswith("M") and "Date" in frame:
            columns = self._sample_columns(frame, int(self.configuration["generation_series_start"]), int(self.configuration["generation_series_stop"]))
            return self._resample_generation(frame, columns, path.name)
        asset = str(self.asset["asset_name"])
        required = ["Year", "Month", "Day", "Hour", asset]
        missing = [column for column in required if column not in frame]
        if missing:
            raise KeyError(f"{path.name} is missing columns: {missing}")
        year_count = self.end.year - self.start.year + 1
        source_year = pd.to_numeric(frame["Year"], errors="coerce")
        work = frame.loc[source_year.between(1, year_count), required].copy()
        boundary = frame.loc[
            source_year.eq(year_count + 1)
            & pd.to_numeric(frame["Month"], errors="coerce").eq(1)
            & pd.to_numeric(frame["Day"], errors="coerce").eq(1)
            & pd.to_numeric(frame["Hour"], errors="coerce").eq(1),
            required,
        ].head(1).copy()
        if boundary.empty:
            boundary = frame.loc[
                source_year.eq(source_year.min())
                & pd.to_numeric(frame["Month"], errors="coerce").eq(1)
                & pd.to_numeric(frame["Day"], errors="coerce").eq(1)
                & pd.to_numeric(frame["Hour"], errors="coerce").eq(1),
                required,
            ].head(1).copy()
        if not boundary.empty:
            boundary["Year"] = year_count + 1
            work = pd.concat([work, boundary], ignore_index=True)
        work["Date"] = pd.to_datetime({"year": self.start.year + work["Year"].astype(int) - 1, "month": work["Month"].astype(int), "day": work["Day"].astype(int), "hour": work["Hour"].astype(int) - 1}, errors="coerce")
        work = work.dropna(subset=["Date"])[["Date", asset]]
        hourly = work.set_index("Date")[asset].sort_index().reindex(pd.date_range(work["Date"].min(), work["Date"].max(), freq="1h")).interpolate(method="time").rename_axis("Date").reset_index()
        return self._resample_generation(hourly, [asset], path.name).rename(columns={asset: "Scenario 1"})

    def load_generation(self, code: str) -> pd.DataFrame:
        path = self.resolve_source(code)
        frame = pd.read_csv(path)
        if code in {"B1S", "B1SM"}:
            return self._load_standard_year(code, path, frame)
        if code.endswith("M"):
            columns = self._sample_columns(frame, int(self.configuration["generation_series_start"]), int(self.configuration["generation_series_stop"]))
            return self._resample_generation(frame, columns, path.name)
        asset = str(self.asset["asset_name"])
        column = next((value for value in (asset, "Generation") if value in frame), None)
        if column is None:
            raise KeyError(f"Neither {asset!r} nor 'Generation' is present in {path.name}.")
        return self._resample_generation(frame, [column], path.name).rename(columns={column: "Scenario 1"})

    def load_lgc(self) -> pd.DataFrame:
        override = float(self.asset.get("LGC_price_override_per_MWh", 0) or 0)
        if override > 0:
            return pd.DataFrame({"Date": self.target_dates, "LGC": override})
        path = self.resolve_source("C1")
        return self._resample(pd.read_csv(path), ["LGC price"], "ffill", path.name).rename(columns={"LGC price": "LGC"})

    def load_tariff(self) -> pd.DataFrame:
        mapping = pd.read_csv(self.resolve_source("D1")).melt(id_vars=["Tariff code", "Day"], var_name="Hour", value_name="ToU_label")
        values = pd.read_csv(self.resolve_source("D2"))
        code = str(self.asset["tariff_code"])
        mapping = mapping.loc[mapping["Tariff code"].astype(str) == code].copy()
        values = values.loc[values["Tariff code"].astype(str) == code].copy()
        if mapping.empty or values.empty:
            raise ValueError(f"Tariff code {code!r} is missing from D1 or D2.")
        mapping["Hour"] = pd.to_numeric(mapping["Hour"], errors="raise").astype(int)
        mapping["Day"] = mapping["Day"].str.lower()
        result = pd.DataFrame({"Date": self.target_dates})
        result["Hour"] = result["Date"].dt.hour + 1
        result["Day"] = np.where(result["Date"].dt.weekday >= 5, "weekend", "weekday")
        result = result.merge(mapping[["Hour", "Day", "ToU_label"]], on=["Hour", "Day"], how="left", validate="many_to_one")
        result = result.merge(values[["ToU", "Import Charge"]], left_on="ToU_label", right_on="ToU", how="left", validate="many_to_one")
        if result[["ToU_label", "Import Charge"]].isna().any().any():
            raise ValueError("D1/D2 do not completely map the configured date range.")
        result["Import_charge"] = pd.to_numeric(result["Import Charge"]) * 100 / 1000
        return result[["Date", "Import_charge", "ToU_label"]]

    def load_physical_constraints(self) -> pd.DataFrame:
        mapping = pd.read_csv(self.resolve_source("E1"))
        technology = str(self.asset["BESS_technology"])
        mapping = (mapping.melt(id_vars=["Project", "Field"], var_name="Year", value_name="Value").assign(Year=lambda value: pd.to_numeric(value["Year"], errors="coerce")).loc[lambda value: value["Project"].astype(str) == technology].pivot(index="Year", columns="Field", values="Value").sort_index().dropna(how="all"))
        required = {"AC inverter", "DC block RTE", "Aux charge", "Aux discharge", "BESS storage degradation"}
        coupling = str(self.asset["BESS_coupling_type"])
        if coupling == "DC":
            required.add("DC converter")
        missing = sorted(required.difference(mapping.columns))
        if mapping.empty or missing:
            raise ValueError(f"E1 has no complete {technology!r} mapping; missing {missing}.")
        year_count = self.end.year - self.start.year + 2
        mapping = mapping.reindex(range(year_count)).ffill()
        annual = pd.DataFrame({"Date": [self.start + pd.DateOffset(years=year) for year in range(year_count)]})
        annual["bess_deg"] = mapping["BESS storage degradation"].to_numpy()
        if coupling == "AC":
            annual["hte_from_grid"] = mapping["DC block RTE"].to_numpy() * mapping["AC inverter"].to_numpy() * mapping["Aux charge"].to_numpy()
            annual["hte_from_pv"] = annual["hte_from_grid"] / (0.9966 * 0.9999)
            annual["hte2grid"] = mapping["Aux discharge"].to_numpy() * mapping["AC inverter"].to_numpy()
            annual["pv_bess_grid"] = annual["hte_from_pv"] * annual["hte2grid"] * mapping["AC inverter"].to_numpy() / (0.9966 * 0.9999)
        else:
            annual["hte_from_pv"] = mapping["DC block RTE"].to_numpy() * mapping["DC converter"].to_numpy() * mapping["Aux charge"].to_numpy() / mapping["AC inverter"].to_numpy()
            annual["hte_from_grid"] = annual["hte_from_pv"] * mapping["AC inverter"].to_numpy() ** 2
            annual["hte2grid"] = mapping["DC converter"].to_numpy() * mapping["Aux discharge"].to_numpy() * mapping["AC inverter"].to_numpy()
            annual["pv_bess_grid"] = annual["hte_from_pv"] * annual["hte2grid"] * mapping["AC inverter"].to_numpy()
        annual["grid_bess_grid"] = annual["hte_from_grid"] * annual["hte2grid"]
        columns = ["bess_deg", "hte_from_pv", "hte_from_grid", "hte2grid", "pv_bess_grid", "grid_bess_grid"]
        return self._resample(annual, columns, "interpolate", "E1")

    def load_bess_sizes(self, code: str) -> pd.DataFrame:
        columns = ["BESS_Duration_h", "BESS_Power_MW", "BESS_MWh", "BESS_MWh_per_interval"]
        if code == "F1":
            duration, power = float(self.asset["BESS_duration_hours"]), float(self.asset["BESS_power_MW"])
            usable_fraction = float(self.asset["BESS_usable_fraction"])
            energy_override = float(self.asset.get("BESS_energy_override_MWh", 0) or 0)
            energy = energy_override if energy_override > 0 else duration * power * usable_fraction
            return pd.DataFrame([{"BESS_Duration_h": duration, "BESS_Power_MW": power, "BESS_MWh": energy, "BESS_MWh_per_interval": power * self.granularity / 60}])
        path = self.resolve_source("F1M")
        result = pd.read_csv(path)
        missing = [column for column in columns if column not in result]
        if missing:
            raise KeyError(f"{path.name} is missing columns: {missing}")
        result = result[columns].apply(pd.to_numeric, errors="coerce")
        start, stop = int(self.configuration["bess_series_start"]), int(self.configuration["bess_series_stop"])
        result = result.iloc[start:stop].reset_index(drop=True)
        if result.empty or result.isna().any().any():
            raise ValueError(f"{path.name} has no valid BESS designs in range [{start}, {stop}).")
        return result

    def _price_inputs(self, selected_code: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        predicted = self.load_price(selected_code)
        if selected_code in {"A2", "A2M"}:
            actual = self.load_price("A1M" if selected_code.endswith("M") else "A1")
            if list(actual.columns) != list(predicted.columns):
                raise ValueError("A1/A2 scenario columns do not match.")
            return actual, predicted
        return predicted.copy(), predicted

    @staticmethod
    def _write(frame: pd.DataFrame, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)

    def load_selected_sources(self, save: bool = True) -> LoadedSources:
        price_code, generation_code, bess_code = (str(self.configuration[name]) for name in ("price_source", "generation_source", "bess_source"))
        price_actual, price_predicted = self._price_inputs(price_code)
        generation, bess_sizes = self.load_generation(generation_code), self.load_bess_sizes(bess_code)
        lgc, tariff, physical = self.load_lgc(), self.load_tariff(), self.load_physical_constraints()
        dataset = pd.DataFrame({"Date": self.target_dates})
        for source in (lgc, tariff, physical):
            dataset = dataset.merge(source, on="Date", how="left", validate="one_to_one")
        if dataset.isna().any().any():
            raise ValueError("The assembled static optimizer dataset contains missing values.")
        if save:
            actual_price_code = "A1M" if price_code == "A2M" else "A1" if price_code == "A2" else price_code
            outputs = [
                (price_actual, f"{source_stem(actual_price_code, self.asset)}.csv"),
                (price_predicted, f"{source_stem(price_code, self.asset)}.csv"),
                (generation, f"{source_stem(generation_code, self.asset)}.csv"),
                (bess_sizes, f"{source_stem(bess_code, self.asset)}.csv"),
                (lgc, f"{source_stem('C1', self.asset)}.csv"),
                (physical, f"{source_stem('E1', self.asset)}.csv"),
                (dataset, "Dataset/dataset.csv"),
            ]
            # A1/A3 use the same frame as actual and predicted price; avoid a
            # redundant second write when both resolve to the same filename.
            written = set()
            for frame, relative in outputs:
                if relative in written:
                    continue
                self._write(frame, PROCESSED_DATA_DIR / relative)
                written.add(relative)
        return LoadedSources(dataset, price_actual, price_predicted, generation, bess_sizes, lgc, tariff, physical)


def load_active_runtime() -> dict:
    """Return flattened runtime values from the active CSV records."""
    runtime = CsvTableStore().runtime()
    configuration, asset = runtime["configuration"], runtime["asset"]
    configuration["price_source"] = route_price(configuration)
    configuration["generation_source"] = route_generation(configuration)
    configuration["bess_source"] = route_bess(configuration)
    granularity = int(configuration["optimization_granularity_in_minutes"])
    perfect_price_foresight = (
        configuration["price_timeframe"] == "Historical back-test"
        and configuration["price_foresight"] == "Perfect foresight"
    )
    transform = (
        "Normal predicted value"
        if perfect_price_foresight
        else str(configuration["prediction_signal_transform"])
    )
    predicted_price_type = {"Normal predicted value": 1, "Transform to binary signal": 2, "Both": 3}[transform]
    run_multiple = any(_bool(configuration[field]) for field in ("price_run_synthetic_distribution", "generation_run_synthetic_distribution", "bess_run_synthetic_distribution"))
    return {
        "grid_import_penalty": float(asset["grid_import_penalty"]),
        "project_poi_export_limit_mwh_per_interval": float(asset["project_poi_export_limit_MW"]) * granularity / 60,
        "project_poi_import_limit_mwh_per_interval": float(asset["project_poi_import_limit_MW"]) * granularity / 60,
        "demand_charge_off_peak": float(asset["demand_charge_off_peak"]),
        "demand_charge_shoulder": float(asset["demand_charge_shoulder"]),
        "demand_charge_peak": float(asset["demand_charge_peak"]),
        "export_charge_sun_soaker": float(asset["export_charge_sun_soaker"]),
        "optimize_with_duos": _bool(asset["optimize_with_DUOS"]),
        "tlf_pv_injection": float(asset["TLF_PV_injection"]),
        "tlf_bess_injection": float(asset["TLF_BESS_injection"]),
        "tlf_bess_withdrawal": float(asset["TLF_BESS_withdrawal"]),
        "price_escalation_factor": float(asset["price_escalation_factor"]),
        "lgc_eligibility_basis": str(asset["LGC_eligibility_basis"]),
        "start_soc": float(asset["BESS_Start_SoC"]), "cycles_per_day": float(asset["BESS_cycles_per_day"]),
        "operation_granularity_in_minutes": granularity,
        "optimization_avoid_edge_effect_total_hours": float(configuration["optimization_avoid_edge_effect_hours"]),
        "number_of_intervals_per_window": int(configuration["number_of_intervals_per_window"]),
        "curtailment_penalty": 0.0 if perfect_price_foresight else float(configuration["curtailment_penalty"]),
        "trade_signal_penalty": 0.0 if perfect_price_foresight else float(configuration["trade_signal_penalty"]),
        "execution_mode": configuration["execution_mode"],
        "display_window_scheduler_visual": _bool(configuration["display_window_scheduler_visual"]),
        "prediction_signal_transform": 1 if transform == "Normal predicted value" else 2,
        "predicted_price_type": predicted_price_type,
        "optimisation_directive": 1 if configuration["price_foresight"] == "Perfect foresight" else 2,
        "run_multiple_optimisations": run_multiple,
    }


class ConfigurationEditor:
    """ipywidgets CRUD editor for both CSV tables."""

    def __init__(self, store: CsvTableStore | None = None):
        try:
            import ipywidgets as widgets
        except ImportError as exc:
            raise ImportError("Install ipywidgets (included in requirements.txt) to use the editor.") from exc
        self.w = widgets
        self.store = store or CsvTableStore()
        self.status = widgets.Output()
        self.asset_widgets = self._asset_widgets()
        self.configuration_widgets = self._configuration_widgets()
        self.widget = widgets.VBox([self._tabs(), self.status])
        self._refresh_selectors()

    def _row(self, label, widget):
        return self.w.HBox([self.w.Label(label, layout=self.w.Layout(width="290px")), widget])

    def _section(self, title, fields):
        heading = self.w.HTML(f"<h4 style='background:#ddd;padding:6px'>{title}</h4>")
        return self.w.VBox([heading, *[self._row(label, widget) for label, widget in fields]])

    def _asset_widgets(self):
        w = self.w
        fields = {
            "record": w.Dropdown(description="Record"),
            "asset_name": w.Text(), "state": w.Dropdown(options=STATES),
            "tariff_code": w.Dropdown(options=self.store.tariff_codes()),
            "demand_charge_off_peak": w.FloatText(disabled=True),
            "demand_charge_shoulder": w.FloatText(disabled=True),
            "demand_charge_peak": w.FloatText(disabled=True),
            "export_charge_sun_soaker": w.FloatText(disabled=True),
            "project_poi_export_limit_MW": w.FloatText(),
            "project_poi_import_limit_MW": w.FloatText(),
            "project_poi_export_limit_MWh_per_interval": w.FloatText(disabled=True),
            "project_poi_import_limit_MWh_per_interval": w.FloatText(disabled=True),
            "grid_import_penalty": w.FloatText(),
            "optimize_with_DUOS": w.Checkbox(value=True),
            "TLF_PV_injection": w.FloatText(value=1.0),
            "TLF_BESS_injection": w.FloatText(value=1.0),
            "TLF_BESS_withdrawal": w.FloatText(value=1.0),
            "price_escalation_factor": w.FloatText(value=1.0),
            "LGC_eligibility_basis": w.Dropdown(options=LGC_ELIGIBILITY_BASES),
            "LGC_price_override_per_MWh": w.FloatText(value=0.0),
            "BESS_technology": w.Dropdown(options=self.store.bess_technologies()),
            "BESS_coupling_type": w.Dropdown(options=("AC", "DC")),
            "BESS_duration_hours": w.FloatText(), "BESS_power_MW": w.FloatText(),
            "BESS_usable_fraction": w.BoundedFloatText(min=0.000001, max=1, step=0.01, value=1.0),
            "BESS_energy_override_MWh": w.FloatText(value=0.0),
            "BESS_Start_SoC": w.BoundedFloatText(min=0, max=1, step=0.05),
            "BESS_cycles_per_day": w.FloatText(), "BESS_lifetime_years": w.IntText(),
        }
        fields["record"].observe(self._load_asset, names="value")
        fields["tariff_code"].observe(self._update_asset_autos, names="value")
        return fields

    def _configuration_widgets(self):
        w = self.w
        fields = {
            "record": w.Dropdown(description="Record"), "configuration_name": w.Text(),
            "active": w.Checkbox(value=True), "asset": w.Dropdown(),
            "execution_mode": w.Dropdown(options=("sequential", "parallel")),
            "display_window_scheduler_visual": w.Checkbox(value=True),
            "optimization_start_date": w.Text(placeholder="DD/MM/YYYY HH:MM"),
            "optimization_end_date": w.Text(placeholder="DD/MM/YYYY HH:MM"),
            "optimization_granularity_in_minutes": w.Dropdown(options=(5, 10, 15, 30, 60), value=60),
            "dataset_date_range": w.Text(disabled=True),
            "dataset_year_month_mapping": w.Text(disabled=True),
            "dataset_index_range": w.Text(disabled=True),
            "optimization_horizon_hours": w.FloatText(value=744),
            "optimization_avoid_edge_effect_hours": w.FloatText(value=24),
            "number_of_intervals_per_window": w.IntText(disabled=True),
            "price_timeframe": w.Dropdown(options=PRICE_TIMEFRAMES),
            "price_foresight": w.Dropdown(options=FORESIGHT_TYPES),
            "price_run_synthetic_distribution": w.Checkbox(),
            "price_series_range": w.IntRangeSlider(min=0, max=1000, value=(0, 100), continuous_update=False),
            "trade_signal_penalty": w.FloatText(value=1),
            "prediction_signal_transform": w.Dropdown(options=PREDICTION_TRANSFORMS),
            "curtailment_penalty": w.FloatText(),
            "generation_timeframe": w.Dropdown(options=PRICE_TIMEFRAMES),
            "generation_foresight": w.Dropdown(options=FORESIGHT_TYPES),
            "generation_profile": w.Dropdown(options=GENERATION_PROFILES),
            "generation_run_synthetic_distribution": w.Checkbox(),
            "generation_series_range": w.IntRangeSlider(min=0, max=1000, value=(0, 100), continuous_update=False),
            "bess_run_synthetic_distribution": w.Checkbox(),
            "bess_series_range": w.IntRangeSlider(min=0, max=1000, value=(0, 10), continuous_update=False),
            "price_source": w.Text(disabled=True), "generation_source": w.Text(disabled=True),
            "bess_source": w.Text(disabled=True),
        }
        fields["record"].observe(self._load_configuration, names="value")
        for name in (
            "price_timeframe", "price_foresight", "price_run_synthetic_distribution",
            "generation_timeframe", "generation_foresight", "generation_profile",
            "generation_run_synthetic_distribution", "bess_run_synthetic_distribution",
            "optimization_start_date", "optimization_end_date",
            "optimization_granularity_in_minutes", "optimization_horizon_hours",
            "optimization_avoid_edge_effect_hours",
        ):
            fields[name].observe(self._update_configuration_choices, names="value")
        return fields

    def _tabs(self):
        w, a, c = self.w, self.asset_widgets, self.configuration_widgets
        asset_save = w.Button(description="Save asset", button_style="success")
        asset_save.on_click(self._save_asset)
        asset_page = w.VBox([
            a["record"],
            self._section("General details", [(name, a[name]) for name in ("asset_name", "state", "tariff_code")]),
            self._section("Automatic tariff values", [(name, a[name]) for name in ("demand_charge_off_peak", "demand_charge_shoulder", "demand_charge_peak", "export_charge_sun_soaker")]),
            self._section("Grid and settlement information", [(name, a[name]) for name in ("project_poi_export_limit_MW", "project_poi_import_limit_MW", "project_poi_export_limit_MWh_per_interval", "project_poi_import_limit_MWh_per_interval", "grid_import_penalty", "optimize_with_DUOS", "TLF_PV_injection", "TLF_BESS_injection", "TLF_BESS_withdrawal", "price_escalation_factor", "LGC_eligibility_basis", "LGC_price_override_per_MWh")]),
            self._section("BESS information", [(name, a[name]) for name in ("BESS_technology", "BESS_coupling_type", "BESS_duration_hours", "BESS_power_MW", "BESS_usable_fraction", "BESS_energy_override_MWh", "BESS_Start_SoC", "BESS_cycles_per_day", "BESS_lifetime_years")]),
            asset_save,
        ])
        config_save = w.Button(description="Save configuration", button_style="success")
        config_save.on_click(self._save_configuration)
        config_page = w.VBox([
            c["record"],
            self._section("General details", [(name, c[name]) for name in ("configuration_name", "active", "asset")]),
            self._section("Code execution", [(name, c[name]) for name in ("execution_mode", "display_window_scheduler_visual")]),
            self._section("Experiment timeframes", [(name, c[name]) for name in ("optimization_start_date", "optimization_end_date", "optimization_granularity_in_minutes", "dataset_date_range", "dataset_year_month_mapping", "dataset_index_range")]),
            self._section("LP mechanics", [(name, c[name]) for name in ("optimization_horizon_hours", "optimization_avoid_edge_effect_hours", "number_of_intervals_per_window")]),
            self._section("Electricity price decisions", [(name, c[name]) for name in ("price_timeframe", "price_foresight", "price_run_synthetic_distribution", "price_series_range", "trade_signal_penalty", "prediction_signal_transform", "curtailment_penalty", "price_source")]),
            self._section("Electricity generation decisions", [(name, c[name]) for name in ("generation_timeframe", "generation_foresight", "generation_profile", "generation_run_synthetic_distribution", "generation_series_range", "generation_source")]),
            self._section("BESS size decisions", [(name, c[name]) for name in ("bess_run_synthetic_distribution", "bess_series_range", "bess_source")]),
            config_save,
        ])
        tabs = w.Tab(children=[asset_page, config_page])
        tabs.set_title(0, "Assets")
        tabs.set_title(1, "Configurations")
        return tabs

    def _refresh_selectors(self):
        assets = self.store.assets()["asset_name"].astype(str).tolist()
        configurations = self.store.configurations()["configuration_name"].astype(str).tolist()
        self.asset_widgets["record"].options = [("Create new…", ""), *[(name, name) for name in assets]]
        self.configuration_widgets["record"].options = [("Create new…", ""), *[(name, name) for name in configurations]]
        self.configuration_widgets["asset"].options = assets
        if assets and not self.configuration_widgets["asset"].value:
            self.configuration_widgets["asset"].value = assets[0]

    def _load_asset(self, change):
        if not change["new"]:
            return
        row = self.store.assets().loc[lambda value: value["asset_name"].astype(str) == str(change["new"])]
        if row.empty:
            return
        record = row.iloc[0].to_dict()
        for name, widget in self.asset_widgets.items():
            if name == "record" or name not in record:
                continue
            try:
                if isinstance(widget, (self.w.FloatText, self.w.BoundedFloatText)):
                    widget.value = float(record[name])
                elif isinstance(widget, self.w.IntText):
                    widget.value = int(float(record[name]))
                else:
                    widget.value = record[name]
            except (TypeError, ValueError):
                pass

    def _update_asset_autos(self, _=None):
        code = self.asset_widgets["tariff_code"].value
        if code:
            for name, value in self.store._tariff_autos(code).items():
                self.asset_widgets[name].value = value

    def _save_asset(self, _):
        values = {name: widget.value for name, widget in self.asset_widgets.items() if name != "record"}
        with self.status:
            self.status.clear_output()
            try:
                record = self.store.save_asset(values)
                self._refresh_selectors()
                self.asset_widgets["record"].value = record["asset_name"]
                print(f"Saved asset {record['asset_name']!r}.")
            except Exception as exc:
                print(f"Asset not saved: {exc}")

    def _configuration_values(self):
        result = {name: widget.value for name, widget in self.configuration_widgets.items() if name != "record"}
        result["price_series_start"], result["price_series_stop"] = result.pop("price_series_range")
        result["generation_series_start"], result["generation_series_stop"] = result.pop("generation_series_range")
        result["bess_series_start"], result["bess_series_stop"] = result.pop("bess_series_range")
        return result

    def _load_configuration(self, change):
        if not change["new"]:
            return
        row = self.store.configurations().loc[lambda value: value["configuration_name"].astype(str) == str(change["new"])]
        if row.empty:
            return
        record = row.iloc[0].to_dict()
        ranges = {
            "price_series_range": (int(float(record["price_series_start"])), int(float(record["price_series_stop"]))),
            "generation_series_range": (int(float(record["generation_series_start"])), int(float(record["generation_series_stop"]))),
            "bess_series_range": (int(float(record["bess_series_start"])), int(float(record["bess_series_stop"]))),
        }
        for name, widget in self.configuration_widgets.items():
            if name == "record":
                continue
            value = ranges.get(name, record.get(name))
            if value is None:
                continue
            try:
                if isinstance(widget, self.w.Checkbox):
                    widget.value = _bool(value)
                elif name == "optimization_granularity_in_minutes":
                    widget.value = int(float(value))
                elif isinstance(widget, self.w.FloatText):
                    widget.value = float(value)
                else:
                    widget.value = value
            except (TypeError, ValueError):
                pass
        self._update_configuration_choices()

    def _update_configuration_choices(self, _=None):
        c = self.configuration_widgets
        if c["price_timeframe"].value == "Future projection":
            c["price_foresight"].options = ("Not applicable",)
            c["price_foresight"].disabled = True
        else:
            c["price_foresight"].options = FORESIGHT_TYPES
            c["price_foresight"].disabled = False
        perfect_price_foresight = (
            c["price_timeframe"].value == "Historical back-test"
            and c["price_foresight"].value == "Perfect foresight"
        )
        for name in ("trade_signal_penalty", "prediction_signal_transform", "curtailment_penalty"):
            c[name].disabled = perfect_price_foresight
        if c["generation_timeframe"].value == "Future projection":
            c["generation_foresight"].options = ("Not applicable",)
            c["generation_foresight"].disabled = True
            c["generation_profile"].options = ("Not applicable",)
            c["generation_profile"].disabled = True
        else:
            c["generation_foresight"].options = FORESIGHT_TYPES
            c["generation_foresight"].disabled = False
            if c["generation_foresight"].value == "Perfect foresight":
                c["generation_profile"].options = GENERATION_PROFILES
                c["generation_profile"].disabled = False
            else:
                c["generation_profile"].options = ("Not applicable",)
                c["generation_profile"].disabled = True
        c["price_series_range"].disabled = not c["price_run_synthetic_distribution"].value
        c["generation_series_range"].disabled = not c["generation_run_synthetic_distribution"].value
        c["bess_series_range"].disabled = not c["bess_run_synthetic_distribution"].value
        values = self._configuration_values()
        c["price_source"].value = route_price(values)
        c["generation_source"].value = route_generation(values)
        c["bess_source"].value = route_bess(values)
        try:
            derived = self.store._derived_configuration(values)
            for name in ("dataset_date_range", "dataset_year_month_mapping", "dataset_index_range", "number_of_intervals_per_window"):
                c[name].value = derived[name]
        except Exception:
            pass

    def _save_configuration(self, _):
        with self.status:
            self.status.clear_output()
            try:
                record = self.store.save_configuration(self._configuration_values())
                self._refresh_selectors()
                self.configuration_widgets["record"].value = record["configuration_name"]
                print(f"Saved configuration {record['configuration_name']!r}. Restart the optimizer kernel to reload it.")
            except Exception as exc:
                print(f"Configuration not saved: {exc}")

    def display(self):
        from IPython.display import display
        display(self.widget)
        return self
