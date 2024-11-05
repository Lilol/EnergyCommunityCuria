import logging

from pandas import merge, concat, DataFrame

from data_processing_pipeline.pipeline_stage import PipelineStage


logger = logging.getLogger(__name__)


class DataMerger(PipelineStage):
    @staticmethod
    def check_labels(label, *data):
        if not all(label in dat for dat in data):
            raise ValueError(f"Not all data have a '{label}' label")

        # Extract 'label' columns and compare
        label_series = [df[label] for df in data]
        all_match = len(set.intersection(*map(set, label_series))) == len(label_series[0])

        return {'all_match': all_match,
            'matching_dataframes': [i + 1 for i, match in enumerate(all_match) for _ in range(match)]}

    def merge(self, to_merge, merge_on, *args, **kwargs):
        raise NotImplementedError('"merge" method in DataMerger base class is not implemented.')


class DataframeMerger(DataMerger):
    def merge(self, to_merge, merge_on, *args, **kwargs):
        # Check if the same labels are present across all data supplied
        matches = self.check_labels(merge_on, to_merge)
        if not matches["all_match"]:
            logger.warning(f"Labels in '{merge_on}' does not match in all dataframes.")

        merged = to_merge[0]
        for merge_right in to_merge[1:]:
            merged = merge([merged, merge_right], on=merge_on)

        return merged


class DatasetConcat(DataMerger):
    def merge(self, to_merge, merge_on, *args, **kwargs):
        axis = kwargs.get('axis', 0)
        all_data = DataFrame()
        for df in to_merge:
            all_data = concat((all_data, df), axis=axis)
        return all_data
