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

4. Set up a virtual envrionment based on the **requirements.txt** file specifying the packages
5. Run the script **preprocessing.py**

### Parametric evaluation
#### The script **parametric_evaluation\run_parametric_evaluation.py** is responsible for the following:
1. Dataset creation in the form of `xarray` dataarray
    Reads input data for different user types (PV plants, families, users) and stores it in an xarray dataarray.
    Transforms it by aggregating monthly and time-of-use (ToU) profiles.
    Combines data into two main datasets:
        `tou_months` — ToU-separated energy data per month.
        `energy_year` — Annual energy data.

2. Metric calculation
The script iterates over different parametric scenarios (varying number of families and battery sizes), and calculates 
various physical, environmental and economic metrics, as well as simulates battery energy storage system (BESS) operations.

3. Visualization - Creates various plots for basic checks of the calculation process

4. Output .csv files of the calculated metrics.

The script automates the entire process and scales the data for multiple users or families.

#### How to run **run_parametric_evaluation.py**?
1. Checkout the repository into project root: **<root>**
2. Copy the entire content of **<root>//config//example_config.ini** to create **<root>//config//config.ini**
3. Set the path of the **input and output files**
4. Set `[parametric_evaluation] evaluation_parameters` parameters in **config.ini**:

   There are three ways to set the parameters:
   * Set battery sizes and number of families separately, the script will iterate over all possible combinations:
   `evaluation_parameters={'bess_sizes': [1,2], 'number_of_families': [20,50,70]}`
   * Specify battery sizes, and for each size, the number of families to evaluate for this specific battery size:
   `evaluation_parameters = {'bess_sizes': {0: [2,3,4], 1: [20,24]}}`
   * Specify number of families and for each, the battery sizes to evaluation
   `evaluation_parameters = {'number_of_families': {20: [10,20,40], 50: [60,9]}}`

5. Set `[parametric_evaluation] to_evaluate` in **config.ini**:
It is a list of metrics to evaluate, the list can contain the following values:
   * physical: self-consumption, self-sufficiency
   * economic: capex, opex
   * environmental: emissions savings ratio, baseline emissions, total emissions
   * self_consumption_targets: Calculates the number of families necessary for reaching a predefined self-consumption target
   * time_aggregation: Calculates physical metrics for various time-aggregations, like year, month, day, ...
   * all: constitutes physical,economic,environmental

6. Set the cost of equipment (in Euros) and emission factors in the input .csv files in the `data` folder:
   * `data\cost_of_equipment.csv`
   * `data\emission_factors.csv`
   
7. Set up a virtual environment based on the **requirements.txt** file specifying the packages
8. Run `parametric_evaluation\run_parametric_evaluation.py`

### Renewable energy community evaluation