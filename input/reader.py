import logging
import os
from os.path import join
from typing import Iterable

import xarray as xr
from pandas import DataFrame, read_csv, concat

from data_processing_pipeline.definitions import Stage
from data_processing_pipeline.pipeline_stage import PipelineStage
from data_storage.dataset import OmnesDataArray
from input.definitions import ColumnName, PvDataSource
from utility import configuration

logger = logging.getLogger(__name__)


class Reader(PipelineStage):
    stage = Stage.READ
    _column_names = {}
    _name = "reader"

    def __init__(self, name=_name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self._directory = "."
        self._filename = ""
        self._data = OmnesDataArray()

    def execute(self, dataset: OmnesDataArray, *args, **kwargs) -> OmnesDataArray:
        municipality = kwargs.pop("municipality", configuration.config.get("rec", "municipalities"))
        municipalities = filter(lambda x: os.path.isdir(join(self._directory, x)), os.listdir(self._directory)) if municipality == "all" else (
            municipality if isinstance(municipality, Iterable) else (municipality,))
        data = OmnesDataArray(data=None, dims=("dim_0","dim_1"), coords={"dim_0": [], "dim_1": []})
        for m in municipalities:
            data = xr.concat([data, self._read_municipality(m)], dim="dim_1")
        return data

    def _read_municipality(self, municipality):
        self._data = read_csv(os.path.join(self._directory, municipality, self._filename), sep=';',
                              usecols=list(self._column_names.keys())).rename(columns=self._column_names)
        self._data.insert(1, ColumnName.MUNICIPALITY, municipality)
        self._data.reset_index(drop=True, inplace=True)
        return OmnesDataArray(data=self._data)


class ProductionDataReader(Reader):
    _name = "production_reader"

    @classmethod
    def create(cls, *args, **kwargs):
        source = configuration.config.get("production", "estimator")
        return PvgisReader(*args, **kwargs) if source == PvDataSource.PVGIS else PvsolReader(*args, **kwargs)


class PvgisReader(ProductionDataReader):
    _name = "pvgis_reader"

    def execute(self, dataset: OmnesDataArray, *args, **kwargs) -> OmnesDataArray:
        municipality = kwargs.pop("municipality", "")
        user = kwargs.pop("user", "")
        return OmnesDataArray(
            read_csv(join(self._directory, municipality, PvDataSource.PVSOL.value, f"{user}.csv"), sep=';', index_col=0,
                     parse_dates=True, date_format="%d/%m/%Y %H:%M"))


class PvsolReader(ProductionDataReader):
    _name = "pvsol_reader"
    _production_column_name = 'Grid Export '  # hourly production of the plants (kW)

    def execute(self, dataset: OmnesDataArray, *args, **kwargs) -> OmnesDataArray:
        municipality = kwargs.pop("municipality", "")
        user = kwargs.pop("user", "")
        production = read_csv(join(self._directory, municipality, PvDataSource.PVSOL.value, f"{user}.csv"), sep=';',
                              decimal=',', low_memory=False, skiprows=range(1, 17), index_col=0, header=0,
                              parse_dates=True, date_format="%d.%m. %H:%M",
                              usecols=["Time", self._production_column_name])

        days = production[self._production_column_name].groupby(production.index.dayofyear)
        production = DataFrame(data=[items.values for g, items in days], index=days.groups.keys(),
                               columns=days.groups[1] - days.groups[1][0])
        return OmnesDataArray(data=production)


class PvPlantReader(Reader):
    _name = "pvplant_reader"

    _column_names = {'pod': ColumnName.USER,  # code or name of the associated end user
                     'descrizione': ColumnName.DESCRIPTION,  # description of the associated end user
                     'indirizzo': ColumnName.USER_ADDRESS,  # address of the end user
                     'pv_size': ColumnName.POWER,  # size of the plant (kW)
                     'produzione annua [kWh]': ColumnName.ANNUAL_ENERGY,  # annual energy produced (kWh)
                     'rendita specifica [kWh/kWp]': ColumnName.ANNUAL_YIELD,  # specific annual production (kWh/kWp)
                     }

    def __init__(self, name="pv_plant_reader", *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self._data_source = configuration.config.get("production", "estimator")
        self._directory = "DatiComuni"
        self._filename = "lista_impianti.csv"  # list of plants
        self._production_data_reader = ProductionDataReader.create()
        self._production_data = DataFrame()

    def execute(self, dataset: OmnesDataArray, *args, **kwargs) -> OmnesDataArray:
        municipality = kwargs.pop("municipality")
        super().execute(dataset, *args, **kwargs)
        for user in self._data[ColumnName.USER].unique():
            production = self._production_data_reader.execute(municipality=municipality, user=user)
            production[ColumnName.USER] = user
            self._production_data = concat([production, self._production_data], axis="rows")
        return OmnesDataArray(data=self._data)


class UsersReader(Reader):
    _name = "users_reader"
    _column_names = {'pod': ColumnName.USER,  # code or name of the end user
                     'descrizione': ColumnName.DESCRIPTION,  # description of the end user
                     'indirizzo': ColumnName.USER_ADDRESS,  # address of the end user
                     'tipo': ColumnName.USER_TYPE,  # type of end user
                     'potenza': ColumnName.POWER,  # maximum available power of the end-user (kW)
                     }

    def __init__(self, name=_name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self._directory = "DatiComuni"
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

    def __init__(self, name=_name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self._directory = "DatiComuni"
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
            self._data.columns.isin(self._time_of_use_energy_column_names.values())])
        return OmnesDataArray(data=self._data)


class GlobalConstReader(Reader):
    def execute(self, dataset: OmnesDataArray, *args, **kwargs) -> OmnesDataArray:
        self._data = read_csv(join(self._directory, self._filename), sep=';', index_col=0, header=0).rename(
            columns=self._column_names)
        return OmnesDataArray(data=self._data)


class TariffReader(GlobalConstReader):
    _name = "tariff_reader"

    def __init__(self, name=_name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        # ARERA's day-types depending on subdivision into tariff time-slots
        # NOTE : f : 1 - tariff time-slot F1, central hours of work-days
        #            2 - tariff time-slot F2, evening of work-days, and saturdays
        #            3 - tariff times-lot F2, night, sundays and holidays
        self._directory = "Common"
        self._filename = "arera.csv"


class TypicalLoadProfileReader(GlobalConstReader):
    _name = "typical_load_profile_reader"
    _column_names = {'type': ColumnName.USER_TYPE,  # code or name of the end user
                     'month': ColumnName.MONTH}

    # Reference profiles from GSE
    def __init__(self, name=_name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self._directory = "Common"
        self._filename = "y_ref_gse.csv"
