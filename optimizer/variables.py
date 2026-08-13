import pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parent

BESS_SIZING = {
    "bess_mw": 4.96, # Power rating in MW.
    "bess_hours": 2.0, # Nominal duration in hours. Formula: bess_mwh / bess_mw # Note: Approximate value under OEM specification: 9.38197356150388 / 4.96 ~= 1.89152693
    "bess_hours_usable_fraction": 0.94576346, # Usable fraction of nominal energy capacity (0-1).
    "bess_hours_usable": 1.89152693,  # Usable duration in hours. # Formula: bess_hours * bess_hours_usable_fraction
    "oem_provided_mwh": 9.38197356150388, # Authoritative engineering/OEM usable-capacity value in MWh.
    "option": 1,  # bess_mwh calculation selector.  # 1: use OEM value, 2: recompute from fraction, 3: recompute from usable hours
    "bess_technology": "Jinko", # BESS technology used by optimization/degradation assumptions.
}

BESS_DECISION_CONTROL_VARIABLES = {
    "start_soc": 0.5, # Starting state of charge (0-1).
    "cycles_per_day": 1.0, # Allowed equivalent full cycles per day.
    "bess_lifetime_years": 20,  # Number of years of BESS operation before replacement or end-of-life.
}

LOCATION_SPECIFIC_VARIABLES_GRID = {
    "grid_import_penalty": 1, # $/MWh
    "project_poi_export_limit_mw": 4.99, 
    "project_poi_import_limit_mw": 4.2,
}

LOCATION_SPECIFIC_VARIABLES_GEOGRAPHICAL = {
    "province": "NSW", # Unique to price data source
    "site": "Orange 2B", # Unique to generation data source
    "tariff_code": "BHND4LS", # Unique to tariff data source
}

SIMULATION_CONTROL_VARIABLES_TIME = {
    "operation_start_date": "01/01/2026 00:00",
    "operation_end_date": "31/12/2056 23:59",
    "operation_granularity_in_minutes": 60,
    "optimization_horizon_hours": 31*24, # was 744
    "optimization_avoid_edge_effect_total_hours": 24, # was 24
}

SIMULATION_CONTROL_VARIABLES_PROTOCOL = {
    "curtailment_penalty": 0.0,
    "run_multiple_optimisations": True,
    "execution_mode": "parallel", # ["sequential", "parallel"]
    "display_window_scheduler_visual": False,  # Optimise using: Actual electricity price -> 1, Predicted electricity price -> 2
    "optimisation_directive": 1,
}


# Settings related to predicted electricity price
if SIMULATION_CONTROL_VARIABLES_PROTOCOL["optimisation_directive"] == 2:
    
    # Use predicted prices - > 1,  # Use direct bess instructions -> 2, # Use both -> 3
    predicted_price_type = 1
    
    if predicted_price_type == 2 or predicted_price_type == 3:
        trade_signal_penalty = 1.0

    HEURISTIC_MODEL_SETTINGS = {
        "operating_protocol_seasonal_grouping_granularity": "14D", # [18H,14D,MS,QS,YS] How often the signals change
        "operating_protocol_transform": "None", # ["None","Smoothed","Pessimistic","Pessimistic_Smoothed"]
        "operating_protocol_window_size": 2, # How many neighbouring intervals get averaged into the resulting intervals
        "apply_blend": False
    }

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------
# Required calculations
# -------------------------------------------------------------------------------------------------------------------------------------

# > Formula: bess_mw × bess_hours × bess_hours_usable_fraction
if BESS_SIZING["option"] == 1:
    BESS_SIZING["bess_mwh"] = BESS_SIZING["oem_provided_mwh"]
elif BESS_SIZING["option"] == 2:
    BESS_SIZING["bess_mwh"] = BESS_SIZING["bess_mw"] * (BESS_SIZING["bess_hours"] * BESS_SIZING["bess_hours_usable_fraction"])
elif BESS_SIZING["option"] == 3:
    BESS_SIZING["bess_mwh"] = BESS_SIZING["bess_mw"] * BESS_SIZING["bess_hours_usable"]
else:
    raise ValueError("BESS_SIZING['option'] must be 1, 2, or 3")


# Backward-compatible module-level names for variables.x access from other files.
for _cfg in (
    BESS_SIZING,
    BESS_DECISION_CONTROL_VARIABLES,
    LOCATION_SPECIFIC_VARIABLES_GRID,
    LOCATION_SPECIFIC_VARIABLES_GEOGRAPHICAL,
    SIMULATION_CONTROL_VARIABLES_TIME,
    SIMULATION_CONTROL_VARIABLES_PROTOCOL,
):
    globals().update(_cfg)


_granularity = SIMULATION_CONTROL_VARIABLES_TIME["operation_granularity_in_minutes"]

bess_energy_limit_per_interval_mw = BESS_SIZING["bess_mw"] / (60 / _granularity)
bess_energy_limit_per_interval_mwh = BESS_SIZING["bess_mwh"] / (60 / _granularity)

project_poi_import_limit_mwh_per_interval = LOCATION_SPECIFIC_VARIABLES_GRID["project_poi_import_limit_mw"] / (60 / _granularity)
project_poi_export_limit_mwh_per_interval = LOCATION_SPECIFIC_VARIABLES_GRID["project_poi_export_limit_mw"] / (60 / _granularity)
number_of_intervals_per_window = int(
    (
        SIMULATION_CONTROL_VARIABLES_TIME["optimization_horizon_hours"]
        + SIMULATION_CONTROL_VARIABLES_TIME["optimization_avoid_edge_effect_total_hours"]
    )
    * (60 / _granularity)
)

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------
# Load lookup values
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------
Network_tariff_values = pd.read_csv( ROOT / "1_Dataset/1_Raw_data/Network_tariff_values.csv")
physical_constraint = pd.read_csv( ROOT / "1_Dataset/2_Processed_data/physical_constraint.csv")

_tariff_code = LOCATION_SPECIFIC_VARIABLES_GEOGRAPHICAL["tariff_code"]

demand_charge_off_peak = (Network_tariff_values[(Network_tariff_values["Tariff code"] == _tariff_code) &(Network_tariff_values["ToU"] == "Off-peak")]["Demand charge"].iloc[-1]) * 1000
demand_charge_shoulder = (Network_tariff_values[(Network_tariff_values["Tariff code"] == _tariff_code) &(Network_tariff_values["ToU"] == "Shoulder")]["Demand charge"].iloc[-1]) * 1000
demand_charge_peak = (Network_tariff_values[(Network_tariff_values["Tariff code"] == _tariff_code) &(Network_tariff_values["ToU"] == "Peak")]["Demand charge"].iloc[-1]) * 1000
export_charge_sun_soaker = (Network_tariff_values[(Network_tariff_values["Tariff code"] == _tariff_code) &(Network_tariff_values["ToU"] == "Sun Soaker")]["Export Charge"].iloc[-1]) * 1000
discharge_efficiency = physical_constraint["hte2grid"].iloc[0]

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------
# Calculated required reference arrays 
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------
dataset_date_range = pd.DataFrame(pd.date_range(start = pd.to_datetime(SIMULATION_CONTROL_VARIABLES_TIME["operation_start_date"],dayfirst=True),end = pd.to_datetime(SIMULATION_CONTROL_VARIABLES_TIME["operation_end_date"],dayfirst=True),freq=f"{_granularity}min"),columns=['Date'])
dataset_year_month_mapping = sorted(set(zip(dataset_date_range['Date'].dt.year,dataset_date_range['Date'].dt.month)))
dataset_index_range = list(range(len(dataset_date_range)))





