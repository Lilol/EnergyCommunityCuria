import logging
import os
from os.path import join

import numpy as np
import xarray as xr
from pandas import read_csv, read_excel

from data_processing_pipeline.definitions import Stage
from data_processing_pipeline.pipeline_stage import PipelineStage
from data_storage.data_store import DataStore
from data_storage.dataset import OmnesDataArray
from input.definitions import DataKind, PvDataSource
from utility import configuration
from utility.enum_definitions import convert_enum_to_value, convert_value_to_enum

logger = logging.getLogger(__name__)


class Read(PipelineStage):
    stage = Stage.READ
    _column_names = {}
    _name = "reader"
    _input_root = configuration.config.get("path", "input")
    _directory = ""
    filename = ""

    def __init__(self, name=_name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self._path = join(self._input_root, self._directory)
        self._filename = kwargs.pop("filename", self.filename)
        self._data = OmnesDataArray()

    def execute(self, dataset: OmnesDataArray, *args, **kwargs) -> OmnesDataArray:
        municipalities = kwargs.pop("municipality", configuration.config.get("rec", "municipalities"))
        self._data = xr.concat([self._read_municipality(m) for m in municipalities], dim=DataKind.MUNICIPALITY.value)
        return self._data

    def _read_municipality(self, municipality):
        data = read_csv(os.path.join(self._path, municipality, self._filename), sep=';',
                        usecols=list(self._column_names.keys())).rename(columns=self._column_names)
        return OmnesDataArray(data=data.reset_index(drop=True)).expand_dims(DataKind.MUNICIPALITY.value).assign_coords(
            {DataKind.MUNICIPALITY.value: [municipality]})


class ReadProduction(Read):
    _name = "production_reader"

    def __init__(self, name=_name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self._pv_resource_reader = ReadPvProcuction.create()

    def execute(self, dataset: OmnesDataArray, *args, **kwargs) -> OmnesDataArray:
        municipalities = kwargs.pop("municipality", configuration.config.get("rec", "municipalities"))
        user_data = DataStore()["pv_plants"]
        self._data = xr.concat(
            [self._pv_resource_reader.execute(self._data, municipality=municipality, user=user) for municipality in
             municipalities for user in
             user_data.sel({DataKind.MUNICIPALITY.value: municipality})[DataKind.USER.value].values],
            dim=DataKind.USER.value)
        return self._data


class ReadPvProcuction(Read):
    _name = "pvresource_reader"
    _directory = "DatiComuni"

    @classmethod
    def create(cls, *args, **kwargs):
        source = configuration.config.get("production", "estimator")
        return ReadPvgis(*args, **kwargs) if source == PvDataSource.PVGIS else ReadPvSol(*args, **kwargs)


class ReadPvgis(ReadPvProcuction):
    _name = "pvgis_reader"
    _production_column_name = 'Irradiance onto horizontal plane '  # hourly production of the plants (kW)
    _column_names = {"power": DataKind.POWER}

    def execute(self, dataset: OmnesDataArray, *args, **kwargs) -> OmnesDataArray:
        municipality = kwargs.pop("municipality", configuration.config.get("rec", "location"))
        user = kwargs.pop("user")
        production = read_csv(join(self._path, municipality, PvDataSource.PVGIS.value, f"{user}.csv"), sep=';',
                              index_col=0, parse_dates=True, date_format="%d/%m/%Y %H:%M").rename(
            columns=self._column_names)
        production = OmnesDataArray(data=production).rename(
            {"dim_1": DataKind.POWER.value, "timestamp": DataKind.TIME.value}).expand_dims(
            [DataKind.MUNICIPALITY.value, DataKind.USER.value]).assign_coords(
            {DataKind.MUNICIPALITY.value: [municipality, ], DataKind.USER.value: [user, ]}).squeeze(
            DataKind.POWER.value, drop=True)
        return production


class ReadPvSol(ReadPvProcuction):
    _name = "pvsol_reader"
    _production_column_name = 'Grid Export '  # hourly production of the plants (kW)
    _column_names = {_production_column_name: DataKind.POWER}

    def execute(self, dataset: OmnesDataArray, *args, **kwargs) -> OmnesDataArray:
        municipality = kwargs.pop("municipality", configuration.config.get("rec", "location"))
        user = kwargs.pop("user", "")
        production = read_csv(join(self._path, municipality, PvDataSource.PVSOL.value, f"{user}.csv"), sep=';',
                              decimal=',', skiprows=range(1, 17), index_col=0, header=0, parse_dates=True,
                              date_format="%d.%m. %H:%M", usecols=["Time", self._production_column_name]).rename(
            columns=self._column_names)
        ref_year = configuration.config.getint("time", "year")
        production.index = production.index.map(lambda x: x.replace(year=ref_year))
        production = OmnesDataArray(data=production).rename(
            {"dim_1": DataKind.POWER.value, "Time": DataKind.TIME.value}).expand_dims(
            [DataKind.MUNICIPALITY.value, DataKind.USER.value]).assign_coords(
            {DataKind.MUNICIPALITY.value: [municipality, ], DataKind.USER.value: [user, ]}).squeeze(
            DataKind.POWER.value, drop=True)
        return production


class ReadPvPlantData(Read):
    _name = "pv_plant_reader"

    _column_names = {'pod': DataKind.USER,  # code or name of the associated end user
                     'descrizione': DataKind.DESCRIPTION,  # description of the associated end user
                     'indirizzo': DataKind.USER_ADDRESS,  # address of the end user
                     'pv_size': DataKind.POWER,  # size of the plant (kW)
                     'produzione annua [kWh]': DataKind.ANNUAL_ENERGY,  # annual energy produced (kWh)
                     'rendita specifica [kWh/kWp]': DataKind.ANNUAL_YIELD,  # specific annual production (kWh/kWp)
                     }

    _directory = "DatiComuni"
    filename = "lista_impianti.csv"

    def __init__(self, name=_name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)


class ReadUserData(Read):
    _name = "users_reader"
    _column_names = {'pod': DataKind.USER,  # code or name of the end user
                     'descrizione': DataKind.DESCRIPTION,  # description of the end user
                     'indirizzo': DataKind.USER_ADDRESS,  # address of the end user
                     'tipo': DataKind.USER_TYPE,  # type of end user
                     'potenza': DataKind.POWER,  # maximum available power of the end-user (kW)
                     }

    _directory = "DatiComuni"
    filename = "lista_pod.csv"  # list of end-users

    def __init__(self, name=_name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)


class ReadBills(Read):
    _name = "bill_reader"
    _time_of_use_energy_column_names = {f'f{i}': f"{DataKind.TOU_ENERGY.value}{i}" for i in range(1,
                                                                                                  configuration.config.getint(
                                                                                                      "tariff",
                                                                                                      "number_of_time_of_use_periods") + 1)}

    _column_names = {'pod': DataKind.USER,  # code or name of the end user
                     'anno': DataKind.YEAR,  # year
                     'mese': DataKind.MONTH,  # number of the month
                     'totale': DataKind.ANNUAL_ENERGY,  # Annual consumption
                     'f0': DataKind.MONO_TARIFF, **_time_of_use_energy_column_names}

    _directory = "DatiComuni"
    filename = "dati_bollette.csv"

    def __init__(self, name=_name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)

    def execute(self, dataset: OmnesDataArray, *args, **kwargs) -> OmnesDataArray:
        super().execute(dataset, *args, **kwargs)
        # Check that each user has exactly 12 rows in the bills dataframe
        users = self._data.sel({"dim_1": DataKind.USER})
        if not (np.all(users.groupby(users).count() == 12)).all():
            logger.warning(
                "All end users in 'data_users_bills' must have exactly 12 rows, but a user is found with more or less"
                " rows.")
        # Time of use labels
        configuration.config.set_and_check("tariff", "time_of_use_labels", self._data["dim_1"][
            self._data["dim_1"].isin(list(self._time_of_use_energy_column_names.values()))].values,
                                           setter=configuration.config.setarray, check=False)
        return self._data


class ReadCommonData(Read):
    def execute(self, dataset: OmnesDataArray, *args, **kwargs) -> OmnesDataArray:
        self._data = read_csv(join(self._path, self._filename), sep=';', index_col=0, header=0).rename(
            columns=self._column_names)
        return OmnesDataArray(data=self._data)


class ReadTariff(ReadCommonData):
    # ARERA's division depending on subdivision into tariff time-slots
    # NOTE : f : 1 - tariff time-slot F1, central hours of work-days
    #            2 - tariff time-slot F2, evening of work-days, and saturdays
    #            3 - tariff times-lot F2, night, sundays and holidays
    _name = "tariff_reader"
    _directory = "Common"
    filename = "arera.csv"

    def __init__(self, name=_name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)


class ReadTypicalLoadProfile(ReadCommonData):
    _name = "typical_load_profile_reader"
    _column_names = {'type': DataKind.USER_TYPE,  # code or name of the end user
                     'month': DataKind.MONTH}
    _directory = "Common"
    filename = "y_ref_gse.csv"

    # Reference profiles from GSE
    def __init__(self, name=_name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)


class ReadGseDatabase(ReadTypicalLoadProfile):
    _name = "gse_database_reader"
    _column_names = convert_value_to_enum
    _directory = "DatabaseGSE"
    filename = "gse_ref_profiles.xlsx"

    # Reference profiles from GSE
    def __init__(self, name=_name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)

    def execute(self, dataset: OmnesDataArray, *args, **kwargs) -> OmnesDataArray:
        self._data = read_excel(join(self._path, self._filename), index_col=0, header=0, parse_dates=True,
                                date_format="%d.%m.%Y %H.%M").rename(columns=lambda x: ReadGseDatabase._column_names(x))
        return OmnesDataArray(data=self._data)
