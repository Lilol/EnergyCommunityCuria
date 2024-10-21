import logging
import os
from os.path import join

from pandas import DataFrame, concat, read_csv, timedelta_range, to_datetime, date_range

import configuration
from input.definitions import InputColumn, PvDataSource

logger = logging.getLogger(__name__)


class Reader:
    column_names = {}
    fig_check = configuration.config.getboolean("visualization", "check_by_plotting")

    def __init__(self, *args, **kwargs):
        self._directory = "."
        self._filename = ""
        self._data = DataFrame()

    def read(self, *args, **kwargs):
        municipality = kwargs.pop("municipality", "")
        input_path = os.path.join(self._directory, municipality, self._filename)
        self._data = read_csv(input_path, sep=';', usecols=list(self.column_names.keys())).rename(
            columns=self.column_names)
        self._data.insert(1, InputColumn.MUNICIPALITY, municipality)
        self._data.reset_index(drop=True, inplace=True)


class ProductionDataReader(Reader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def create(cls, *args, **kwargs):
        source = configuration.config.get("production", "estimator")
        return PvgisReader(*args, **kwargs) if source == PvDataSource.PVGIS else PvsolReader(*args, **kwargs)


class PvgisReader(ProductionDataReader):
    def read(self, *args, **kwargs):
        municipality = kwargs.pop("municipality", "")
        user = kwargs.pop("user", "")
        return read_csv(join(self._directory, municipality, PvDataSource.PVSOL.value, f"{user}.csv"), sep=';',
                        index_col=0, parse_dates=True, date_format="%d/%m/%Y %H:%M")


class PvsolReader(ProductionDataReader):
    production_column_name = 'Grid Export '  # hourly production of the plants (kW)

    def read(self, *args, **kwargs):
        municipality = kwargs.pop("municipality", "")
        user = kwargs.pop("user", "")
        production = read_csv(join(self._directory, municipality, PvDataSource.PVSOL.value, f"{user}.csv"), sep=';',
                              decimal=',', low_memory=False, skiprows=range(1, 17), index_col=0, header=0,
                              parse_dates=True, date_format="%d.%m. %H:%M",
                              usecols=["Time", self.production_column_name])

        days = production[self.production_column_name].groupby(production.index.dayofyear)
        production = DataFrame(data=[items.values for g, items in days], index=days.groups.keys(),
                               columns=days.groups[1] - days.groups[1][0])
        return production


class PvPlantReader(Reader):
    column_names = {'pod': InputColumn.USER,  # code or name of the associated end user
                    'descrizione': InputColumn.DESCRIPTION,  # description of the associated end user
                    'indirizzo': InputColumn.USER_ADDRESS,  # address of the end user
                    'pv_size': InputColumn.POWER,  # size of the plant (kW)
                    'produzione annua [kWh]': InputColumn.ANNUAL_ENERGY,  # annual energy produced (kWh)
                    'rendita specifica [kWh/kWp]': InputColumn.ANNUAL_YIELD,  # specific annual production (kWh/kWp)
                    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._data_source = configuration.config.get("production", "estimator")
        self._directory = "DatiCommuni"
        self._filename = "lista_impianti.csv"  # list of plants
        self._production_data_reader = ProductionDataReader.create()
        self._production_data = DataFrame()

    def read(self, *args, **kwargs):
        municipality = kwargs.get("municipality", "")
        super().read(*args, **kwargs)
        for user in self._data[InputColumn.USER].unique():
            production = self._production_data_reader.read(municipality=municipality, user=user)
            self._production_data = self.create_yearly_profile(production, self._production_data, user)

    @classmethod
    def create_yearly_profile(cls, profile, all_profiles, user_name=None):
        profile = DataFrame(data=profile, columns=timedelta_range(start="0 Days", freq="1h", periods=profile.shape[1]))

        ref_year = configuration.config.getint("time", "year")
        profile.index = to_datetime(date_range(start=f"{ref_year}-01-01", end=f"{ref_year}-12-31", freq="d"))
        profile[InputColumn.USER] = user_name
        profile[InputColumn.YEAR] = profile.index.year
        profile[InputColumn.MONTH] = profile.index.month
        profile[InputColumn.DAY_OF_MONTH] = profile.index.day
        profile[InputColumn.WEEK] = profile.index.dt.isocalendar().week
        profile[InputColumn.SEASON] = profile.index.month % 12 // 3 + 1
        profile[InputColumn.DAY_OF_WEEK] = profile.index.dayofweek
        return concat((all_profiles, profile), axis=0)


class UsersReader(Reader):
    column_names = {'pod': InputColumn.USER,  # code or name of the end user
                    'descrizione': InputColumn.DESCRIPTION,  # description of the end user
                    'indirizzo': InputColumn.USER_ADDRESS,  # address of the end user
                    'tipo': InputColumn.USER_TYPE,  # type of end user
                    'potenza': InputColumn.POWER,  # maximum available power of the end-user (kW)
                    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._directory = "DatiCommuni"
        self._filename = "lista_pod.csv"  # list of end-users


class BillsReader(Reader):
    column_names = {'pod': InputColumn.USER,  # code or name of the end user
                    'anno': InputColumn.YEAR,  # year
                    'mese': InputColumn.MONTH,  # number of the month
                    'f0': InputColumn.TOU_ENERGY, 'f1': InputColumn.TOU_ENERGY, 'f2': InputColumn.TOU_ENERGY,
                    'f3': InputColumn.TOU_ENERGY,  # ToU monthly consumption (kWh)
                    'totale': InputColumn.ANNUAL_ENERGY,  # Annual consumption
                    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._directory = "DatiCommuni"
        self._filename = "dati_bollette.csv"  # monthly consumption data

    def read(self, *args, **kwargs):
        super().read(*args, **kwargs)
        # Check that each user has exactly 12 rows in the bills dataframe
        if not (self._data[InputColumn.USER].value_counts() == 12).all():
            logger.warning(
                "All end users in 'data_users_bills' must have exactly 12 rows, but a user is found with more or less rows.")
