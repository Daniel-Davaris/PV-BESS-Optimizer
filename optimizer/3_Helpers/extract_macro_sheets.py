import pandas as pd

def load_ref():
    SHEET_MAPPINGS = {
        'Forecast curves': 'Forecast curves.csv',
        'Macro curves': 'Macro curves.csv',
        'NUOS': 'Network_time_of_use_tariff_mapping.csv',
        'NUOS2': 'Network_tariff_values.csv',
        'RTE2': 'Physical_constraints_mapping.csv'
    }

    macro_data = pd.read_excel("../1_Dataset/1_Raw_data/Archive/aus_macro_v2.xlsx", sheet_name=list(SHEET_MAPPINGS.keys()))

    for sheet_name, df in macro_data.items():
        output_path = f"{SHEET_MAPPINGS[sheet_name]}"
        df.to_csv(output_path, index=False)
        
load_ref()