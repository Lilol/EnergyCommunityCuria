# 🌞 Energy Community Curia

A framework for evaluating energy communities using household consumption, PV generation, tariffs, and batteries.

---

## ⚙️ Setup Guide

### 🔍 Preprocessing

#### 📜 What `preprocessing.py` Does:

1. **Reads and processes input data** — household data, PV generation, tariffs, bills.
2. **Validates and transforms** the data — cleaning and converting to `xarray.DataArray` structures.
3. **Stores processed data** in structured `xarray` format.
4. **Generates visualizations** for load and generation.
5. **Writes outputs** — including cleaned load/generation/user data as `.csv` and `.nc` files.

> 📌 This script automates preprocessing for multiple families and users.

#### ▶️ How to Run `preprocessing.py`

1. Clone the repo to your project root: `<root>`
2. Copy the default config:
   ```
   cp <root>/config/example_config.ini <root>/config/config.ini
   ```
3. Set the input/output file paths in `config.ini`:
   ```ini
   [path]
   root=path/to/files
   ```

4. Ensure this folder structure under the root:

   ```
   input/
   ├── common/
   │   ├── arera.csv
   │   └── y_ref_gse.csv
   ├── DatabaseGSE/
   │   └── gse_ref_profiles.csv
   └── DatiComuni/
       └── SettimoTorinese/
           ├── PVGIS/ <generator_production_files>
           ├── PVSOL/ <generator_production_files>
           ├── bollette_domestici.csv
           ├── dati_bollette.csv
           ├── lista_impianti.csv
           └── lista_pod.csv
   ```

5. Outputs will be created at:
   ```
   output/DatiProcessati/SettimoTorinese/
   ├── Loads/ <loads_by_ids>
   ├── Generators/ <generators_by_ids>
   ├── data_users.csv / data_plants.csv / ...
   └── families_<num_families>.csv
   ```

6. Set up your Python environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # or .venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```

7. Run the script:
   ```bash
   python preprocessing.py
   ```

---

### 📊 Parametric Evaluation

#### 📜 What `run_parametric_evaluation.py` Does:

1. **Creates datasets** in `xarray` format:
   - Aggregates monthly and ToU (time-of-use) data
   - Outputs:
     - `tou_months`: ToU-separated monthly data
     - `energy_year`: Yearly energy data

2. **Computes metrics**:
   - Physical, environmental, economic metrics
   - Simulates battery energy storage systems (BESS)

3. **Visualizes** the evaluation process.

4. **Exports results** to `.csv`

#### ▶️ How to Run `run_parametric_evaluation.py`

1. Clone the repo to your project root: `<root>`
2. Copy the config:
   ```
   cp <root>/config/example_config.ini <root>/config/config.ini
   ```

3. Set paths in `config.ini`:
   ```ini
   [path]
   root=path/to/files
   ```

4. Define parameters in `[parametric_evaluation]`:
   - Option 1: Cartesian product of battery sizes × families:
     ```ini
     evaluation_parameters = {'bess_sizes': [1, 2], 'number_of_families': [20, 50, 70]}
     ```
   - Option 2: Specific family sets for each battery:
     ```ini
     evaluation_parameters = {'bess_sizes': {0: [2, 3, 4], 1: [20, 24]}}
     ```
   - Option 3: Battery sets for each number of families:
     ```ini
     evaluation_parameters = {'number_of_families': {20: [10, 20, 40], 50: [60, 9]}}
     ```

5. Choose metrics under `to_evaluate`:
   ```ini
   to_evaluate = ['physical', 'economic', 'environmental', 'all']
   ```
   Options include:
   - `physical`: self-consumption, self-sufficiency
   - `economic`: capex, opex
   - `environmental`: baseline emissions, emissions in case of REC establishment, savings
   - `metric_targets`, `time_aggregation`, `all`

6. Define cost/emissions in:
   ```
   data/cost_of_equipment.csv
   data/emission_factors.csv
   ```

7. Set up the environment and run:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   python parametric_evaluation/run_parametric_evaluation.py
   ```

---

### 🌱 Renewable Energy Community Evaluation

---

🛠 Maintained by: Lilla Barancsuk
📬 Contact: https://github.com/Lilol
