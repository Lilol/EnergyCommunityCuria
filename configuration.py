import logging
from configparser import RawConfigParser, ExtendedInterpolation
from os import getcwd
from os.path import join, dirname
from sys import argv


logger = logging.getLogger(__name__)


class ConfigurationManager:
    def __init__(self, config_filename=join(getcwd(), "config", "config.ini")):
        self.__config = RawConfigParser(allow_no_value=True, interpolation=ExtendedInterpolation())
        self.__config.read_file(open(config_filename))
        self._registered_entries = {
            "data":
                {"eval_sample_count": self._process_eval_sample_count},
        }

    def _process_eval_sample_count(self):
        sample_count = self.__config.get("data", "eval_sample_count")
        try:
            return int(sample_count)
        except ValueError:
            return sample_count

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
        self.__config.set(section, key, value)

    def setint(self, section, key, value):
        self.__config.set(section, key, f"{value}")

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


# def do_configuration():
config_file = argv[2] if len(argv) >= 3 else join(dirname(__file__), 'config', 'config.ini')
config = ConfigurationManager(config_filename=config_file)
