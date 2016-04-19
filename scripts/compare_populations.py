#!/usr/bin/python

import pandas as pd
import numpy as np
import os

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

def load_hosp_data(filename):

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

    df = pd.read_csv(filename,
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

    return df

def load_pop_data(filename):

    df = pd.read_csv(filename,
                  sep=',',
                  header=0,
                  #dtype=DTYPES,
                  #na_values={'Excess Readmission Ratio': 'Not Available',
                  #          },
                  #engine='c',
                    )

    # Attempt to coerce to numbers (including strings), with unconvertible
    # values becoming NaN.
    #df = df.convert_objects(convert_numeric=True)

    # remove 'not-availables'
    #df = df.replace('Not Available', np.nan)

    return df

def write_data(filename, df):

    df.to_pickle(filename)

    # what?


def plot_readmission_vs_population(df_hosp, df_pop, filename):

    import matplotlib.pyplot as plt
    import us
    import scipy.stats

    # Plot
    fig, ax = plt.subplots()

    # get the readmission ratio

    # convert state abbreviations to names
    states = df_hosp['State'].unique()
    y = np.empty(len(states), dtype=float)
    x = np.empty(len(states), dtype=float)
    for i in xrange(len(y)):
        state_name = us.states.lookup(states[i]).name
        index = df_pop[df_pop['NAME'] == state_name].index.tolist()
        x[i] = df_pop['POPESTIMATE2015'][index].values[0]

        index = df_hosp[df_hosp['State'] == states[i]].index.tolist()
        median = scipy.stats.nanmedian(df_hosp['Excess Readmission Ratio'][index].values)
        y[i] = median

        #ax.annotate(states[i],xy=(x[i]/1e6,y[i],), xycoords='data',)

    #x = x / 1e6
    ax.scatter(x/1e6,y,)
    #ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
    ax.set_xlabel('Population [1,000,000]')
    ax.set_ylabel('Excess Readmission Ratio')
    x_scalar = np.max(x) * 0.1
    ax.set_xlim([np.min(x)-x_scalar, np.max(x)+x_scalar])
    y_scalar = np.max(y) * 0.1
    ax.set_ylim([np.min(y)-y_scalar, np.max(y)+y_scalar])
    ax.legend()

    plt.savefig(filename, dpi=100)
    plt.show()

def main():

    PLOT = 1
    WRITE = 1

    # Filenames
    DIR_DATA = '../data/'
    DIR_DATA_PROD = '../data_products/'
    DIR_PLOTS = '../plots/'
    FILENAME_DATA_HOSP = DIR_DATA + 'Hospital_Readmissions_Reduction_Program.csv'
    FILENAME_DATA_POP = DIR_DATA + 'NST-EST2015-alldata.csv'
    FILENAME_PLOT_READ_VS_POP = DIR_PLOTS + 'readmission_vs_population'
    FILENAME_PLOT_READ_VS_STATE = DIR_PLOTS + 'readmission_vs_state'

    # Hospital Name,
    # Provider Number,
    # State,
    # Measure Name,Number of
    # Discharges,Footnote,Excess Readmission Ratio,Predicted Readmission
    # Rate,Expected Readmission Rate,Number of Readmissions,Start Date,End Date

    df_hosp = load_hosp_data(FILENAME_DATA_HOSP)
    df_pop = load_pop_data(FILENAME_DATA_POP)



    if PLOT:
        import matplotlib.pyplot as plt

        plot_readmission_vs_population(df_hosp,
                                       df_pop,
                                       FILENAME_PLOT_READ_VS_POP)

        # Plot Readmission ratio box plot with state
        fig, ax = plt.subplots()
        df_hosp.boxplot(column='Excess Readmission Ratio', by='State', ax=ax)
        ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
        plt.savefig(FILENAME_PLOT_READ_VS_STATE, dpi=100)
        plt.show()


if __name__ == '__main__':
    main()

