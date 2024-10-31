from pandas import merge

from data_processing_pipeline.pipeline_stage import PipelineStage


class DataMerger(PipelineStage):
    def merge(self, to_merge, merge_on, *args, **kwargs):
        raise NotImplementedError('"merge" method in DataMerger base class is not implemented.')


class MyMerger(DataMerger):
    def merge(self, to_merge, merge_on, *args, **kwargs):
        # Check if the same labels are present across all data supplied
        self.check_labels(merge_on, to_merge)

        merged = to_merge[0]
        for merge_right in to_merge[1:]:
            merged = merge([merged, merge_right], on=merge_on)

        return merged

    @staticmethod
    def check_labels(label, *data):
        label_columns = [df.columns.get(label, None) for df in data]
        if not all(label in dat for dat in data):
            raise ValueError(f"Not all data have a '{label}' label")

        # Extract 'label' columns
        label_series = [df[label] for df in data]

        # Compare values
        all_match = len(set.intersection(*map(set, label_series))) == len(label_series[0])

        # Return results
        return {'all_match': all_match,
            'matching_dataframes': [i + 1 for i, match in enumerate(all_match) for _ in range(match)]}
