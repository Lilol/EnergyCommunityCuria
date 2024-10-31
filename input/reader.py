import logging
import os
from os.path import join

import numpy as np
from pandas import DataFrame, concat, read_csv, timedelta_range, to_datetime, date_range
from xarray import DataArray

import configuration
from data_processing_pipeline.definitions import Stage
from data_processing_pipeline.pipeline_stage import PipelineStage
from input.definitions import ColumnName, PvDataSource

logger = logging.getLogger(__name__)


class Reader(PipelineStage):
    stage = Stage.READ
    column_names = {}
    fig_check = configuration.config.getboolean("visualization", "check_by_plotting")

    def __init__(self, name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self._directory = "."
        self._filename = ""
        self._data = DataArray()

    def execute(self, *args, **kwargs):
        municipality = kwargs.pop("municipality", "")
        input_path = os.path.join(self._directory, municipality, self._filename)
        self._data = read_csv(input_path, sep=';', usecols=list(self.column_names.keys())).rename(
            columns=self.column_names)
        self._data.insert(1, ColumnName.MUNICIPALITY, municipality)
        self._data.reset_index(drop=True, inplace=True)
        return self._data.to_xarray()


class ProductionDataReader(Reader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def create(cls, *args, **kwargs):
        source = configuration.config.get("production", "estimator")
        return PvgisReader(*args, **kwargs) if source == PvDataSource.PVGIS else PvsolReader(*args, **kwargs)


class PvgisReader(ProductionDataReader):
    def execute(self, *args, **kwargs):
        municipality = kwargs.pop("municipality", "")
        user = kwargs.pop("user", "")
        return read_csv(join(self._directory, municipality, PvDataSource.PVSOL.value, f"{user}.csv"), sep=';',
                        index_col=0, parse_dates=True, date_format="%d/%m/%Y %H:%M").to_xarray()


class PvsolReader(ProductionDataReader):
    production_column_name = 'Grid Export '  # hourly production of the plants (kW)

    def execute(self, *args, **kwargs):
        municipality = kwargs.pop("municipality", "")
        user = kwargs.pop("user", "")
        production = read_csv(join(self._directory, municipality, PvDataSource.PVSOL.value, f"{user}.csv"), sep=';',
                              decimal=',', low_memory=False, skiprows=range(1, 17), index_col=0, header=0,
                              parse_dates=True, date_format="%d.%m. %H:%M",
                              usecols=["Time", self.production_column_name])

        days = production[self.production_column_name].groupby(production.index.dayofyear)
        production = DataFrame(data=[items.values for g, items in days], index=days.groups.keys(),
                               columns=days.groups[1] - days.groups[1][0])
        return production.to_xarray()


class PvPlantReader(Reader):
    column_names = {'pod': ColumnName.USER,  # code or name of the associated end user
                    'descrizione': ColumnName.DESCRIPTION,  # description of the associated end user
                    'indirizzo': ColumnName.USER_ADDRESS,  # address of the end user
                    'pv_size': ColumnName.POWER,  # size of the plant (kW)
                    'produzione annua [kWh]': ColumnName.ANNUAL_ENERGY,  # annual energy produced (kWh)
                    'rendita specifica [kWh/kWp]': ColumnName.ANNUAL_YIELD,  # specific annual production (kWh/kWp)
                    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._data_source = configuration.config.get("production", "estimator")
        self._directory = "DatiCommuni"
        self._filename = "lista_impianti.csv"  # list of plants
        self._production_data_reader = ProductionDataReader.create()
        self._production_data = DataFrame()

    def execute(self, *args, **kwargs):
        municipality = kwargs.get("municipality", "")
        super().execute(*args, **kwargs)
        for user in self._data[ColumnName.USER].unique():
            production = self._production_data_reader.execute(municipality=municipality, user=user)
            self._production_data = self.create_yearly_profile(production, self._production_data, user)
        return self._data.to_xarray()

    # TODO: goes into post-processing class
    @classmethod
    def create_yearly_profile(cls, profile, all_profiles, user_name=None):
        profile = DataFrame(data=profile, columns=timedelta_range(start="0 Days", freq="1h", periods=profile.shape[1]))

        ref_year = configuration.config.getint("time", "year")
        profile.index = to_datetime(date_range(start=f"{ref_year}-01-01", end=f"{ref_year}-12-31", freq="d"))
        profile[ColumnName.USER] = user_name
        profile[ColumnName.YEAR] = profile.index.year
        profile[ColumnName.MONTH] = profile.index.month
        profile[ColumnName.DAY_OF_MONTH] = profile.index.day
        profile[ColumnName.WEEK] = profile.index.dt.isocalendar().week
        profile[ColumnName.SEASON] = profile.index.month % 12 // 3 + 1
        profile[ColumnName.DAY_OF_WEEK] = profile.index.dayofweek
        return concat((all_profiles, profile), axis=0)


class UsersReader(Reader):
    column_names = {'pod': ColumnName.USER,  # code or name of the end user
                    'descrizione': ColumnName.DESCRIPTION,  # description of the end user
                    'indirizzo': ColumnName.USER_ADDRESS,  # address of the end user
                    'tipo': ColumnName.USER_TYPE,  # type of end user
                    'potenza': ColumnName.POWER,  # maximum available power of the end-user (kW)
                    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._directory = "DatiCommuni"
        self._filename = "lista_pod.csv"  # list of end-users


class BillsReader(Reader):
    time_of_use_energy_column_names = {f'f{i}': f"{ColumnName.TOU_ENERGY.value}{i}" for i in
                                       range(configuration.config.getint("tariff", "number_of_time_of_use_periods"))}

    column_names = {'pod': ColumnName.USER,  # code or name of the end user
                    'anno': ColumnName.YEAR,  # year
                    'mese': ColumnName.MONTH,  # number of the month
                    'totale': ColumnName.ANNUAL_ENERGY,  # Annual consumption,
                    **time_of_use_energy_column_names}

    # TODO: store this in a common object, not in BillsReader
    time_of_use_labels = list(time_of_use_energy_column_names.values())

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._directory = "DatiCommuni"
        self._filename = "dati_bollette.csv"  # monthly consumption data

    def execute(self, *args, **kwargs):
        super().execute(*args, **kwargs)
        # Check that each user has exactly 12 rows in the bills dataframe
        if not (self._data[ColumnName.USER].value_counts() == 12).all():
            logger.warning(
                "All end users in 'data_users_bills' must have exactly 12 rows, but a user is found with more or less"
                " rows.")
        return self._data.to_xarray()


class GlobalConstReader(Reader):
    def execute(self, *args, **kwargs):
        self._data = read_csv(join(self._directory, self._filename), sep=';', index_col=0, header=0)
        return self._data.to_xarray()


class TariffReader(GlobalConstReader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # ARERA's day-types depending on subdivision into tariff time-slots
        # NOTE : f : 1 - tariff time-slot F1, central hours of work-days
        #            2 - tariff time-slot F2, evening of work-days, and saturdays
        #            3 - tariff times-lot F2, night, sundays and holidays
        self._directory = "Common"
        self._filename = "arera.csv"

    def process_data(self):
        # total number and list of tariff time-slots (index f)
        fs = self._data.unique()
        nf = len(fs)

        # number of day-types (index j)
        # NOTE : j : 0 - work-days (monday-friday)
        #            1 - saturdays
        #            2 - sundays and holidays
        nj = np.size(self._data, axis=0)
        js = list(range(nj))

        # number of time-steps during each day (index i)
        ni = np.size(self._data, axis=1)

        # total number of time-steps (index h)
        nh = self._data.size

        # time-steps where there is a change of tariff time-slot
        h_switch_arera = list(np.where(np.diff(np.insert(self._data.flatten(), -1, self._data[0, 0])) != 0)[0])


class TypicalLoadProfileReader(GlobalConstReader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._directory = "Common"
        self._filename = "y_ref_gse.csv"

    def execute(self, *args, **kwargs):
        # reference profiles from GSE
        super().execute(*args, **kwargs)
        # TODO: to postprocessing class
        y_ref_gse = {i: row.values for i, row in self._data.set_index([ColumnName.USER_TYPE, ColumnName.MONTH]).iterrows()}
        return y_ref_gse
