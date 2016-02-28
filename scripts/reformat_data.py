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
    PLOT = 0
    WRITE = 1

    # Filenames
    DIR_DATA = '../data/'
    DIR_DATA_PROD = '../data_products/'
    FILENAME_DATA = DIR_DATA + 'Hospital_Readmissions_Reduction_Program.csv'
    FILENAME_REFORMATED = DIR_DATA + 'Reformatted_Hospital_Data.csv'

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

    # Attempt to coerce to numbers (including strings), with unconvertible
    # values becoming NaN.
    df = df.convert_objects(convert_numeric=True)
    df["Start Date"] = pd.to_datetime(
        df["Start Date"]
        )
    df["End Date"] = pd.to_datetime(
        df["End Date"]
        )

    # remove 'not-availables'
    df = df.replace('Not Available', np.nan)

    if PLOT:
        import matplotlib.pyplot as plt

        # Plot
        fig, ax = plt.subplots()

        df.boxplot(column='Excess Readmission Ratio', by='State', ax=ax)

        ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
        ax.legend()

        plt.show()


if __name__ == '__main__':
    main()
