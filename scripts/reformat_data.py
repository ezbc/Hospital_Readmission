#!/usr/bin/python

import pandas as pd
import numpy as np

def plot_data(df):

    pass

def read_data(filename, filetype='csv'):

    if filetype == 'csv':
        df = pd.read_csv(filename,
                         sep=',',
                         header=0,
                         )
    else:
        df = pd.read_pickle(filename,
                            )

    return df

def write_data(filename, df):

    df.to_pickle(filename)

    # what?

def main():

    # Filenames
    DIR_DATA = '../data/'
    DIR_DATA_PROD = '../data_products/'
    FILENAME_DATA = DIR_DATA + 'Hospital_Readmissions_Reduction_Program.csv'

    # Hospital Name,
    # Provider Number,
    # State,
    # Measure Name,Number of
    # Discharges,Footnote,Excess Readmission Ratio,Predicted Readmission
    # Rate,Expected Readmission Rate,Number of Readmissions,Start Date,End Date



    DTYPES={'Hospital Name': str, #
            'Provider Number': int, #
            'State': str, #
            'Measure Name': str, #
            'Number of Discharges': int, #
            'Footnote': str, #
            'Excess Readmission Ratio': np.float64, #
            'Predicted Readmission Ratio': np.float64, #
            'Expected Readmission Rate': np.float64, #
            'Number of Readmissions': int, #
            'Start Date': str, #
            'End Date': str, #
            }

    df = pd.read_csv(FILENAME_DATA,
                  sep=',',
                  header=0,
                  #dtype=DTYPES,
                  #na_values={'Excess Readmission Ratio': 'Not Available',
                  #          },
                  #engine='c',
                    )

    print('Columns and types:')
    print(df.dtypes)

    # Attempt to coerce to numbers (including strings), with unconvertible
    # values becoming NaN.
    df = df.convert_objects(convert_numeric=True)

    if 0:
        print df.columns.values

        print df['Excess Readmission Ratio']

        groups = df.groupby('State')
        print type(groups)


        import matplotlib.pyplot as plt

        # Plot
        fig, ax = plt.subplots()
        ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
        for name, group in groups:
            ax.plot(group['State'], group['Excess Readmission Ratio'], marker='o', linestyle='', ms=12, label=name)
        ax.legend()

        plt.show()
    else:
        import matplotlib.pyplot as plt

        # Plot
        fig, ax = plt.subplots()

        df.boxplot(column='Excess Readmission Ratio', by='State', ax=ax)

        ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
        ax.legend()

        plt.show()


if __name__ == '__main__':
    main()

