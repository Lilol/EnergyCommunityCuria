import logging
import os
from os.path import join

import xarray as xr
from pandas import read_csv

from data_processing_pipeline.definitions import Stage
from data_processing_pipeline.pipeline_stage import PipelineStage
from data_storage.data_store import DataStore
from data_storage.dataset import OmnesDataArray
from input.definitions import ColumnName, PvDataSource
from utility import configuration

logger = logging.getLogger(__name__)


class Reader(PipelineStage):
    stage = Stage.READ
    _column_names = {}
    _name = "reader"
    _input_root = configuration.config.get("path", "input")
    _directory = ""

    def __init__(self, name=_name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self._path = join(self._input_root, self._directory)
        self._filename = ""
        self._data = OmnesDataArray()

    def execute(self, dataset: OmnesDataArray, *args, **kwargs) -> OmnesDataArray:
        municipalities = kwargs.pop("municipality", configuration.config.get("rec", "municipalities"))
        data = xr.concat([self._read_municipality(m) for m in municipalities], dim=ColumnName.MUNICIPALITY.value)
        return data

    def _read_municipality(self, municipality):
        self._data = read_csv(os.path.join(self._path, municipality, self._filename), sep=';',
                              usecols=list(self._column_names.keys())).rename(columns=self._column_names)
        return OmnesDataArray(data=self._data.reset_index(drop=True)).expand_dims(
            ColumnName.MUNICIPALITY.value).assign_coords({ColumnName.MUNICIPALITY.value: [municipality]})


class ProductionReader(Reader):
    _name = "production_reader"

    def __init__(self, name=_name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self._pv_resource_reader = PvResourceReader.create()

    def execute(self, dataset: OmnesDataArray, *args, **kwargs) -> OmnesDataArray:
        municipalities = kwargs.pop("municipality", configuration.config.get("rec", "municipalities"))
        user_data = DataStore()["pv_plants"]
        self._data = xr.concat(
            [self._pv_resource_reader.execute(self._data, municipality=municipality, user=user) for municipality in
             municipalities for user in user_data.sel(
                {ColumnName.USER_DATA.value: ColumnName.USER, ColumnName.MUNICIPALITY.value: municipality}).values],
            dim=ColumnName.USER.value)
        return self._data


class PvResourceReader(Reader):
    _name = "pvresource_reader"
    _directory = "DatiComuni"

    @classmethod
    def create(cls, *args, **kwargs):
        source = configuration.config.get("production", "estimator")
        return PvgisReader(*args, **kwargs) if source == PvDataSource.PVGIS else PvsolReader(*args, **kwargs)


class PvgisReader(PvResourceReader):
    _name = "pvgis_reader"
    _production_column_name = 'Irradiance onto horizontal plane '  # hourly production of the plants (kW)
    _column_names = {"power": ColumnName.POWER}

    def execute(self, dataset: OmnesDataArray, *args, **kwargs) -> OmnesDataArray:
        municipality = kwargs.pop("municipality", configuration.config.get("rec", "location"))
        user = kwargs.pop("user")
        production = read_csv(join(self._path, municipality, PvDataSource.PVGIS.value, f"{user}.csv"), sep=';',
                              index_col=0, parse_dates=True, date_format="%d/%m/%Y %H:%M").rename(
            columns=self._column_names)
        production = OmnesDataArray(data=production).rename(
            {"dim_1": ColumnName.POWER.value, "timestamp": ColumnName.TIME.value}).expand_dims(
            [ColumnName.MUNICIPALITY.value, ColumnName.USER.value]).assign_coords(
            {ColumnName.MUNICIPALITY.value: [municipality, ], ColumnName.USER.value: [user, ]}).squeeze(
            ColumnName.POWER.value, drop=True)
        return production


class PvsolReader(PvResourceReader):
    _name = "pvsol_reader"
    _production_column_name = 'Grid Export '  # hourly production of the plants (kW)
    _column_names = {_production_column_name: ColumnName.POWER}

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
            {"dim_1": ColumnName.POWER.value, "Time": ColumnName.TIME.value}).expand_dims(
            [ColumnName.MUNICIPALITY.value, ColumnName.USER.value]).assign_coords(
            {ColumnName.MUNICIPALITY.value: [municipality, ], ColumnName.USER.value: [user, ]}).squeeze(
            ColumnName.POWER.value, drop=True)
        return production


class PvPlantReader(Reader):
    _name = "pv_plant_reader"

    _column_names = {'pod': ColumnName.USER,  # code or name of the associated end user
                     'descrizione': ColumnName.DESCRIPTION,  # description of the associated end user
                     'indirizzo': ColumnName.USER_ADDRESS,  # address of the end user
                     'pv_size': ColumnName.POWER,  # size of the plant (kW)
                     'produzione annua [kWh]': ColumnName.ANNUAL_ENERGY,  # annual energy produced (kWh)
                     'rendita specifica [kWh/kWp]': ColumnName.ANNUAL_YIELD,  # specific annual production (kWh/kWp)
                     }

    _directory = "DatiComuni"

    def __init__(self, name=_name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self._filename = "lista_impianti.csv"  # list of plants


class UsersReader(Reader):
    _name = "users_reader"
    _column_names = {'pod': ColumnName.USER,  # code or name of the end user
                     'descrizione': ColumnName.DESCRIPTION,  # description of the end user
                     'indirizzo': ColumnName.USER_ADDRESS,  # address of the end user
                     'tipo': ColumnName.USER_TYPE,  # type of end user
                     'potenza': ColumnName.POWER,  # maximum available power of the end-user (kW)
                     }

    _directory = "DatiComuni"

    def __init__(self, name=_name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self._filename = "lista_pod.csv"  # list of end-users


class BillReader(Reader):
    _name = "bill_reader"
    _time_of_use_energy_column_names = {f'f{i}': f"{ColumnName.TOU_ENERGY.value}{i}" for i in range(1,
                                                                                                    configuration.config.getint(
                                                                                                        "tariff",
                                                                                                        "number_of_time_of_use_periods") + 1)}

    _column_names = {'pod': ColumnName.USER,  # code or name of the end user
                     'anno': ColumnName.YEAR,  # year
                     'mese': ColumnName.MONTH,  # number of the month
                     'totale': ColumnName.ANNUAL_ENERGY,  # Annual consumption
                     'f0': ColumnName.MONO_TARIFF, **_time_of_use_energy_column_names}

    _directory = "DatiComuni"

    def __init__(self, name=_name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self._filename = "dati_bollette.csv"  # monthly consumption data

    def execute(self, dataset: OmnesDataArray, *args, **kwargs) -> OmnesDataArray:
        super().execute(dataset, *args, **kwargs)
        # Check that each user has exactly 12 rows in the bills dataframe
        if not (self._data[ColumnName.USER].value_counts() == 12).all():
            logger.warning(
                "All end users in 'data_users_bills' must have exactly 12 rows, but a user is found with more or less"
                " rows.")
        # Time of use labels
        configuration.config.set_and_check("tariff", "time_of_use_labels", self._data.columns[
            self._data.columns.isin(self._time_of_use_energy_column_names.values())],
                                           setter=configuration.config.setarray, check=False)
        return OmnesDataArray(data=self._data)


class GlobalConstReader(Reader):
    def execute(self, dataset: OmnesDataArray, *args, **kwargs) -> OmnesDataArray:
        self._data = read_csv(join(self._path, self._filename), sep=';', index_col=0, header=0).rename(
            columns=self._column_names)
        return OmnesDataArray(data=self._data)


class TariffReader(GlobalConstReader):
    _name = "tariff_reader"
    _directory = "Common"

    def __init__(self, name=_name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        # ARERA's division depending on subdivision into tariff time-slots
        # NOTE : f : 1 - tariff time-slot F1, central hours of work-days
        #            2 - tariff time-slot F2, evening of work-days, and saturdays
        #            3 - tariff times-lot F2, night, sundays and holidays
        self._filename = "arera.csv"


class TypicalLoadProfileReader(GlobalConstReader):
    _name = "typical_load_profile_reader"
    _column_names = {'type': ColumnName.USER_TYPE,  # code or name of the end user
                     'month': ColumnName.MONTH}
    _directory = "Common"

    # Reference profiles from GSE
    def __init__(self, name=_name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self._filename = "y_ref_gse.csv"
