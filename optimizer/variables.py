import pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parent

# BESS sizing
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------

# bess_mw
bess_mw = 4.96

# bess_hours
# > Formuala: f(x) bess_mwh / bess_mw
# > Note: Approximate value under OEM specification: 9.38197356150388 / 4.96 ≈ 1.89152693 
bess_hours = 2.0  

# bess_hours_usable_fraction
# > Note: Fraction/percentage
bess_hours_usable_fraction = 0.94576346

# bess_hours_usable
# > Formuala: f(x) = bess_hours * bess_hours_usable_fraction
bess_hours_usable = 1.89152693 

# oem_provided_mwh
# > Note: The authoritative engineering/OEM usable-capacity value 
oem_provided_mwh = 9.38197356150388

# bess_mwh
# > Formula: bess_mw × bess_hours × bess_hours_usable_fraction
option = 1
if option == 1:
    bess_mwh = oem_provided_mwh
elif option == 2:
    bess_mwh = bess_mw * (bess_hours * bess_hours_usable_fraction)
elif option == 3:
    bess_mwh = bess_mw * bess_hours_usable


# BESS other
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------
bess_technology = "Jinko"
start_soc = 0.5
cycles_per_day = 1.0

# Location specific - grid
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------
project_poi_export_limit_mw = 4.99
project_poi_import_limit_mw = 4.2
grid_import_penalty = 1

# Location specific - geographical
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------
province = "NSW" # Unique to price data source
site = "Orange 2B" # Unique to generation data source
tariff_code = "BHND4LS" # Unique to tarrif data source

# Optimization time ranges
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------
operation_start_date = "01/01/2026 00:00"
operation_end_date = "31/12/2056 23:59"
operation_granularity_in_minutes = 60
optimization_horizon_hours = 31*24 # was 744
optimization_avoid_edge_effect_total_hours = 24 # was 24


# Operational decision settings
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------

curtailment_penalty = 0.0
run_multiple_optimisations = True
execution_mode = "parallel" # ["sequential", "parallel"]
display_window_scheduler_visual = False 

# Price prediction settings
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------
# Optimise using: Actual electricity price -> 1
# Optimize using: Predicted electricity price -> 2
optimisation_directive = 2

# If using predicted prices
if optimisation_directive == 2:
    # Settings related to predicted electricity price

    # Changing the below settings requires re-running notebooks 2 and 4 in processed data
    operating_protocol_seasonal_grouping_granularity = "14D" # [18H,14D,MS,QS,YS] How ofen the signals change
    operating_protocol_transform = "None" # ["None","Smoothed","Pessimistic","Pessimistic_Smoothed"]
    operating_protocol_window_size = 2 # How many neighbouring intervals get averaged into the resulting intervals
    apply_blend = False

    # Use predicted prices - > 1
    # Use direct bess instructions -> 2
    # Use both -> 3
    predicted_price_type = 1

    
    if predicted_price_type == 2 or predicted_price_type == 3:
        trade_signal_penalty = 1.0
    
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------
# Required calculations
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------
bess_energy_limit_per_interval_mw = bess_mw / (60 / operation_granularity_in_minutes)
bess_energy_limit_per_interval_mwh = bess_mwh / (60 / operation_granularity_in_minutes)

project_poi_import_limit_mwh_per_interval = project_poi_import_limit_mw / (60 / operation_granularity_in_minutes)
project_poi_export_limit_mwh_per_interval = project_poi_export_limit_mw / (60 / operation_granularity_in_minutes)
number_of_intervals_per_window = int((optimization_horizon_hours + optimization_avoid_edge_effect_total_hours) * (60 / operation_granularity_in_minutes))

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------
# Load lookup values
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------
Network_tariff_values = pd.read_csv( ROOT / "1_Dataset/1_Raw_data/Network_tariff_values.csv")
physical_constraint = pd.read_csv( ROOT / "1_Dataset/2_Processed_data/physical_constraint.csv")

demand_charge_off_peak = (Network_tariff_values[(Network_tariff_values["Tariff code"] == tariff_code) &(Network_tariff_values["ToU"] == "Off-peak")]["Demand charge"].iloc[-1]) * 1000
demand_charge_shoulder = (Network_tariff_values[(Network_tariff_values["Tariff code"] == tariff_code) &(Network_tariff_values["ToU"] == "Shoulder")]["Demand charge"].iloc[-1]) * 1000
demand_charge_peak = (Network_tariff_values[(Network_tariff_values["Tariff code"] == tariff_code) &(Network_tariff_values["ToU"] == "Peak")]["Demand charge"].iloc[-1]) * 1000
export_charge_sun_soaker = (Network_tariff_values[(Network_tariff_values["Tariff code"] == tariff_code) &(Network_tariff_values["ToU"] == "Sun Soaker")]["Export Charge"].iloc[-1]) * 1000
discharge_efficiency = physical_constraint["hte2grid"].iloc[0]

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------
# Calculated required reference arrays 
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------
dataset_date_range = pd.DataFrame(pd.date_range(start = pd.to_datetime(operation_start_date,dayfirst=True),end = pd.to_datetime(operation_end_date,dayfirst=True),freq=f"{operation_granularity_in_minutes}min"),columns=['Date'])
dataset_year_month_mapping = sorted(set(zip(dataset_date_range['Date'].dt.year,dataset_date_range['Date'].dt.month)))
dataset_index_range = list(range(len(dataset_date_range)))





if False:
    # Print all variable values
    import types
    for k, v in globals().copy().items():
        if k.startswith("__"):
            continue
        if isinstance(v, (pd.DataFrame, types.ModuleType)):
            continue
        if callable(v):
            continue
        print(f"{k:<45} {v}")