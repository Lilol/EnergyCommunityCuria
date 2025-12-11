import logging
import os
from configparser import RawConfigParser, ExtendedInterpolation
from os import getcwd
from os.path import join, dirname
from sys import argv
from typing import Iterable

from io_operation.input.definitions import PvDataSource, DataKind
from operation.definitions import ScalingMethod
from parameteric_evaluation.definitions import ParametricEvaluationType

logger = logging.getLogger(__name__)


class ConfigurationManager:
    def __init__(self, config_filename=join(getcwd(), "..", "config", "config.ini")):
        self.__config = RawConfigParser(allow_no_value=True, interpolation=ExtendedInterpolation())
        self.__config.read_file(open(config_filename))
        self._registered_entries = {"production": {"estimator": self._process_pv_estimator},
                                    "rec": {"municipalities": self._get_municipalities},
                                    "tariff": {"time_of_use_labels": self._get_tou_labels},
                                    "profile": {"scaling_method": self._get_scaling_method},
                                    "parametric_evaluation": {"evaluation_parameters": self._get_parameter_pack,
                                                              "to_evaluate": self._get_metrics_to_evaluate}}

    def _get_metrics_to_evaluate(self):
        parameters = self.getarray("parametric_evaluation", "to_evaluate", dtype=ParametricEvaluationType)
        if ParametricEvaluationType.ALL in parameters:
            parameters.remove(ParametricEvaluationType.ALL)
            parameters = parameters + [ParametricEvaluationType.PHYSICAL_METRICS,
                                       ParametricEvaluationType.ECONOMIC_METRICS,
                                       ParametricEvaluationType.ENVIRONMENTAL_METRICS]
        return parameters

    def _get_parameter_pack(self):
        # Lazy import to avoid circular dependency
        from parameteric_evaluation.parameter_pack import EvaluationParameterPack

        parameters = self.__config.get("parametric_evaluation", "evaluation_parameters")
        try:
            return EvaluationParameterPack(parameters)
        except ValueError:
            return parameters

    def _process_pv_estimator(self):
        pv_data_source = self.__config.get("production", "estimator")
        try:
            return PvDataSource(pv_data_source)
        except ValueError:
            return pv_data_source

    def _get_municipalities(self):
        municipality = self.__config.get("rec", "municipalities")
        input_dir = self.__config.get("path", "rec_data")  # Bit of a hack
        return list(
            filter(lambda x: os.path.isdir(join(input_dir, x)), os.listdir(input_dir))) if municipality == "all" else (
            municipality if isinstance(municipality, Iterable) else (municipality,))

    def _get_scaling_method(self):
        scaling_method = self.__config.get("profile", "scaling_method")
        try:
            return ScalingMethod(scaling_method)
        except ValueError:
            return scaling_method

    def _get_tou_labels(self):
        return self.getarray("tariff", "time_of_use_labels", str, fallback=[f"{DataKind.TOU_ENERGY.value}{i}" for i in
                                                                            range(1, self.getint("tariff",
                                                                                                 "number_of_time_of_use_periods") + 1)])

    def get(self, section, key, fallback=None):
        if section not in self._registered_entries or key not in self._registered_entries[section]:
            return self._get(section, key, fallback)
        return self._registered_entries[section][key]()

    def _get(self, section, key, fallback=None):
        try:
            value = self.__config.get(section, key, fallback=fallback)
        except Exception as e:
            raise KeyError(f"Section '{section}', key '{key}' problem in configuration: '{e}'")
        if value is None:
            raise KeyError(f"Section '{section}', key '{key}' not found in configuration")

        if "," not in value:
            return value

        values = list(filter(len, value.strip('][').split(',')))
        all_integers = all(element.isdigit() for element in values)
        if all_integers:
            return [int(x) for x in values]

        return [v.replace(" ", "") for v in values]

    def set(self, section, key, value):
        if type(value) != str:
            value = str(value)
            logger.warning(f"Configuration is set with a non-string-type value: {value}")
        self.__config.set(section, key, value)

    def setint(self, section, key, value):
        self.__config.set(section, key, f"{value}")

    def setarray(self, section, key, value):
        self.__config.set(section, key, f",".join(f"{val}" for val in value))

    def setboolean(self, section, key, value):
        boolean_str = 'True' if value else 'False'
        self.__config.set(section, key, boolean_str)

    def getboolean(self, section, key, fallback=None):
        return self.__config.getboolean(section, key, fallback=fallback)

    def getint(self, section, key, fallback=None):
        try:
            return self.__config.getint(section, key, fallback=fallback)
        except:
            return self.__config.get(section, key, fallback=fallback)

    def getstr(self, section, key, fallback=None):
        return self.__config.get(section, key, fallback=fallback)

    def getarray(self, section, key, dtype=str, fallback=None):
        val = self._get(section, key, fallback=fallback)
        try:
            return [dtype(v) for v in val]
        except TypeError:
            return [dtype(val), ]

    def getfloat(self, section, key, fallback=None):
        return self.__config.getfloat(section, key, fallback=fallback)

    def has_option(self, section, option):
        return self.__config.has_option(section, option)

    def set_and_check(self, section, key, value, setter=None, check=True):
        if check and setter is None:
            value_cf = self.getfloat(section, key)
            if value != value_cf:
                logger.warning(
                    f"The value of [section={section}, key={key}] set dynamically (value={value}) does not equal "
                    f"the original value from the configuration file (value={value_cf})")
        if setter is None:
            if type(value) == int:
                self.setint(section, key, value)
            else:
                self.set(section, key, value)
        else:
            # Get the setter function passed using the current Configuration object
            getattr(self, setter.__name__)(section, key, value)


# def do_configuration():
# Only use argv[2] if it's actually a file path, not a pytest flag
import os.path
config_file = join(dirname(__file__), '..', 'config', 'config.ini')
if len(argv) >= 3 and not argv[2].startswith('-') and os.path.exists(argv[2]):
    config_file = argv[2]
config = ConfigurationManager(config_filename=config_file)
