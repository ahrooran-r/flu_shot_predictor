import pandas as pd
import numpy as np


class LabelEncoderWithMissingValues:
    """
    The original code is here: https://stackoverflow.com/a/64402046/10582056
    Modified to ignore given columns
    """

    def __init__(self):
        pass

    def categorical_to_numeric(self, dataset, ignored=[]):
        self.dataset = dataset
        self.summary = None
        self.table_encoder = {}

        for index in self.dataset.columns:
            if self.dataset[index].dtypes == 'object' and index not in ignored:
                column_data_frame = pd.Series(self.dataset[index], name='column').to_frame()
                unique_values = pd.Series(self.dataset[index].unique())
                i = 0
                label_encoder = pd.DataFrame({'value_name': [], 'Encode': []})
                while i <= len(unique_values) - 1:
                    if unique_values.isnull()[i]:
                        label_encoder = label_encoder.append({'value_name': unique_values[i], 'Encode': np.nan},
                                                             ignore_index=True)  # np.nan = -1
                    else:
                        label_encoder = label_encoder.append({'value_name': unique_values[i], 'Encode': i},
                                                             ignore_index=True)
                    i += 1

                output = pd.merge(left=column_data_frame, right=label_encoder, how='left', left_on='column',
                                  right_on='value_name')
                self.summary = output[['column', 'Encode']].drop_duplicates().reset_index(drop=True)
                self.dataset[index] = output.Encode
                self.table_encoder.update({index: self.summary})

            else:
                pass

        return self.table_encoder, self.dataset

    def inverse_numeric_to_categorical(self, table_encoder, df):
        dataset = df.copy()
        for column in table_encoder.keys():
            df_column = df[column].to_frame()
            output = pd.merge(left=df_column, right=table_encoder[column], how='left', left_on=column,
                              right_on='Encode')
            df[column] = output.column
        return df
