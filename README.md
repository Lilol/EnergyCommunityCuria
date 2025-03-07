# Energy Community Curia

## Setup guide
### Preprocessing

#### The script **preprocessing.py** is responsible for the following:
1. Reading and processing data – The script reads household data, user bills, PV plant generation, and tariff time slots.

2. Data validation and transformation – The script cleans, checks, and converts the data into data structures used by the framework (e.g., load profiles, yearly consumption).

3. Data storing in xarray format – Processed data (users, PV plants, tariff data) is saved in as an xarray DataArray.

4. Visualization – It generates charts to analyze energy consumption and production.

5. Output writing – A set of output files containing energy consumption, production and user data is created at the end of the process.

The script automates the entire process and can scale data for multiple users or families.

#### How to run **preprocessing.py**?
1. Checkout the repository into project root: **<root>**
2. Copy the entire content of **<root>//config//example_config.ini** to create **<root>//config//config.ini**
3. Set the path of the **input and output files**

`
[path]
root=path\to\files
`
The root directory for input files should contain the following:

path/to/files/\
├─ input/\
│  &ensp;├─common/\
│    &emsp;├─ arera.csv\
│    &emsp;├─ y_ref_gse.csv\
│  &ensp;├─DatabaseGSE/\
│    ├─ gse_ref_profiles.csv\  
│  &ensp;├─DatiComuni/\
│    &emsp;├─ SettimoTorinese/\
│       &emsp;&emsp;├─ PVGIS/\
│            &emsp;&emsp;&emsp;├─ <generator_production_files_by_ids>\
│       &emsp;&emsp;├─ PVSOL/\
│            &emsp;&emsp;&emsp;├─ <generator_production_files_by_ids>\
│       &emsp;&emsp;├─ bollette_domestici.csv\  
│       &emsp;&emsp;├─ dati_bollette.csv\
│       &emsp;&emsp;├─ lista_impianti.csv\
│       &emsp;&emsp;├─ lista_pod.csv\

The process creates the following output directories:

path/to/files/\
├─ output/\
│  &ensp;├─DatiProcessati/\
│       &emsp;├─ SettimoTorinese/\
│            &emsp;&emsp;├─ Loads/\
│               &emsp;&emsp;&emsp;├─ <loads_by_ids>\
│            &emsp;&emsp;├─ Generators/\
│               &emsp;&emsp;&emsp;├─ <generators_by_ids>\
│            &emsp;&emsp;├─ data_plants.csv\
│            &emsp;&emsp;├─ data_plants_tou.csv\
│            &emsp;&emsp;├─ data_plants_year.csv\
│            &emsp;&emsp;├─ data_families_tou.csv\
│            &emsp;&emsp;├─ data_families_year.csv\
│            &emsp;&emsp;├─ data_users.csv\
│            &emsp;&emsp;├─ data_users_tou.csv\
│            &emsp;&emsp;├─ data_users_year.csv\
│            &emsp;&emsp;├─ data_users_bills.csv\
│            &emsp;&emsp;├─ families_<num_families>.csv\


### Parametric evaluation

### Renewable energy community evaluation