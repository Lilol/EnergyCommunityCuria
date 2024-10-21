from pandas import Index


class DataMerger(object):
    def merge(self, to_merge, *args, **kwargs):
        raise NotImplementedError('"merge" method in DataMerger base class is not implemented.')


class MyMerger(DataMerger):
    def check_labels(*dfs, col):
        # Check if all dataframes have a 'label' column
        label_columns = [df.columns.get(col, None) for df in dfs]
        if not all(col is not None for col in label_columns):
            raise ValueError("Not all dataframes have a 'label' column")

        # Extract 'label' columns
        label_series = [df['label'] for df in dfs]

        # Compare values
        all_match = len(set.intersection(*map(set, label_series))) == len(label_series[0])

        # Return results
        return {
            'all_match': all_match,
            'matching_dataframes': [i + 1 for i, match in enumerate(all_match) for _ in range(match)]
        }

    def merge(self, to_merge, *args, **kwargs):
        # merge([data_users, data_bills], InputColumn.USER)
        merge_on = args[0]

        # Check that all end users in the list and in the bill coincide
        assert set(data_users[InputColumn.USER]) == set(
            data_users_bills[InputColumn.USER]), "All end users in 'data_users' must be also in 'data_users_bills."

        # ----------------------------------------------------------------------------
        # Manage the data

        # Add column with total yearly consumption for each end user
        data_users = data_users.merge(
            data_users_bills.groupby(InputColumn.USER)[InputColumn.TOU_ENERGY].sum().sum(axis=1).rename(
                InputColumn.ANNUAL_ENERGY).reset_index(), on=InputColumn.USER)

        # Add column with yearly consumption by ToU tariff for each end user
        for col in cm.cols_tou_energy:
            data_users = data_users.merge(
                data_users_bills.groupby(InputColumn.USER)[col].sum().rename(col).reset_index(),
                on=InputColumn.USER)