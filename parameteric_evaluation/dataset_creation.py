from dataclasses import dataclass

from pandas import to_datetime

from data_processing_pipeline.data_processing_pipeline import DataProcessingPipeline
from data_storage.store_data import Store
from input.definitions import DataKind
from input.read import Read, ReadPvPlantData
from parameteric_evaluation.definitions import ParametricEvaluationType
from parameteric_evaluation.parametric_evaluator import ParametricEvaluator
from transform.combine.combine import ArrayConcat
from transform.transform import TransformCoordinateIntoDimension, Aggregate, Apply, Rename, TransformPvPlantData
from utility import configuration


class DatasetCreatorForParametricEvaluation(ParametricEvaluator):
    _key = ParametricEvaluationType.DATASET_CREATION
    # Setup and data loading
    _input_properties = {"input_root": configuration.config.get("path", "output")}
    _tou_columns = configuration.config.get("tariff", "time_of_use_labels")
    _dimensions_to_rename = {"coordinate": {"dim_1": DataKind.USER}, "to_replace_dimension": "dim_0",
                             "new_dimension": "user"}
    _coords_to_rename = {"dim_1": DataKind.TOU.value, "group": DataKind.MONTH.value}

    @classmethod
    def create_and_run_user_data_processing_pipeline(cls, user_type, input_filename):
        DataProcessingPipeline("read_and_store", workers=(
            Read(name=user_type, filename=input_filename, **cls._input_properties),
            TransformCoordinateIntoDimension(name=user_type, **cls._dimensions_to_rename),
            # manage hourly data, sum all end users / plants
            Aggregate(name=user_type, aggregate_on={"dim_1": DataKind.MONTH}),
            Apply(name=f"{user_type}_tou_cols", operation=lambda x: x.sel({"dim_1": cls._tou_columns})),
            Rename(coords=cls._coords_to_rename), Store(user_type))).execute()

    @classmethod
    def create_and_run_timeseries_processing_pipeline(cls, profile, input_filename):
        DataProcessingPipeline("read_and_store", workers=(
            Read(name=profile, filename=input_filename, **cls._input_properties),
            TransformCoordinateIntoDimension(name=f"transform_{profile}", **cls._dimensions_to_rename),
            # Get total production and consumption data
            # Here we manage monthly ToU values, we sum all end users/plants
            Apply(name=profile,
                  operation=lambda x: x.assign_coords(dim_1=to_datetime(x.dim_1)).sum(DataKind.USER.value)),
            Store(profile))).execute()

    @classmethod
    def create_dataset_for_parametric_evaluation(cls):
        """ Get total consumption and production for all users separated months and time of use
        Create a single dataframe for both production and consumptions
        https://www.arera.it/dati-e-statistiche/dettaglio/analisi-dei-consumi-dei-clienti-domestici
        """

        @dataclass
        class ParametricEvaluationUserType:
            user_type: str
            filename: str
            profile_type: str
            profile_filename: str

        user_types = (ParametricEvaluationUserType("pv_plants", "data_plants_tou", "pv_profiles", "data_plants_year"),
                      ParametricEvaluationUserType("families", "data_families_tou", "family_profiles",
                                                   "data_families_year"),
                      ParametricEvaluationUserType("users", "data_users_tou", "user_profiles", "data_users_year"))
        for user in user_types:
            cls.create_and_run_user_data_processing_pipeline(user.user_type, user.filename)
            cls.create_and_run_timeseries_processing_pipeline(user.profile_type, user.profile_filename)

        # Create a single dataframe for both production and consumption
        ut = [ut.user_type for ut in user_types]
        profile_types = [ut.profile_type for ut in user_types]
        DataProcessingPipeline("concatenate", workers=(
            ArrayConcat(dim=DataKind.USER.value, arrays_to_merge=ut, coords={DataKind.USER.value: ut}),
            Store("tou_months"),
            ArrayConcat(name="merge_profiles", dim=DataKind.USER.value, arrays_to_merge=profile_types),
            Store("energy_year"))).execute()

        DataProcessingPipeline("pv_plants",
                               workers=(ReadPvPlantData(), TransformPvPlantData(), Store("data_plants"))).execute()
