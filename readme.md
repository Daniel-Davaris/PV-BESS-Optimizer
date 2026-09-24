# Optimizer

Assets and optimizer configurations are stored in CSV tables:

- `optimizer/configuration_tables/assets.csv`
- `optimizer/configuration_tables/configurations.csv`

Open `optimizer/1_Dataset/1_load_dynamic_data_sources.ipynb` to create or edit
records with the ipywidgets UI. Exactly one configuration must be active. The
decision fields select A1–A3, B1–B3, and F1 sources according to the process
maps shown in the notebook.

After saving the active configuration, run the notebook's final cell to rebuild
the processed optimizer inputs. Processed source files retain their selected
A/B/C/E/F taxonomy names; generic price, generation, and BESS handoff filenames
are not used. Then run `optimizer/2_Optimizer/1_optimize.ipynb`. The optimizer
reads its runtime settings directly from the active CSV records.

Asset records also contain the settlement and usable-capacity assumptions used
by the optimizer: PV/BESS injection and withdrawal TLFs, the LGC eligibility
basis, and the usable fraction of nominal BESS energy. Historical A1 prices are
treated as interval-ending values when they are resampled. Annual optimizer
results include separate energy revenue, LGC revenue, DUOS charges, and net
merchant revenue columns; monthly DUOS maxima are carried across rolling
windows so a calendar month is charged only once.

Asset-specific benchmark cases can also set a nominal price-escalation factor,
an LGC price override, and an explicit internal BESS-energy override. The
`optimize_with_DUOS` choice controls whether network demand charges influence
dispatch; DUOS is still calculated and reported separately. Results include
both gross merchant revenue and net merchant revenue after DUOS.

Raw source filenames follow the convention documented in the loader notebook.
State tokens are lowercase (for example `..._nsw.csv`) and spaces in asset
tokens are replaced with underscores (for example `..._Orange_2B.csv`).

Raw data, processed datasets, and optimization results are local runtime files
and are excluded from Git. The two small CSV configuration tables above remain
source-controlled and do not require Git LFS.
