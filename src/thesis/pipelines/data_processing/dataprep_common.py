# Import necessary libraries
import math

import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


def load_data_and_save():
    df_freq0 = fetch_openml(data_id=41214, as_frame=True).data
    df_sev0 = fetch_openml(data_id=41215, as_frame=True).data

    return df_freq0, df_sev0


def common_dataprep(df_freq, df_sev0):
    df_freq['ClaimNb'] = df_freq['ClaimNb'].clip(upper=4)  # Schelldorfer
    df_freq['Density'] = df_freq['Density'].apply(lambda x: round(math.log(x), 2))  # Schelldorfer
    df_freq['Area'] = df_freq['Area'].apply(lambda x: ord(x) - 64)  # Schelldorfer

    # Own additions
    df_sev0["ClaimAmount"] = df_sev0["ClaimAmount"].clip(upper=200000)
    df_freq["Frequency"] = df_freq["ClaimNb"] / df_freq["Exposure"]

    # Transforming to correct datatypes
    df_freq["IDpol"] = df_freq["IDpol"].astype(int)
    df_sev0["IDpol"] = df_sev0["IDpol"].astype(int)

    # Merging severity and frequency datasets -> Severity dataset
    df_sev = pd.merge(df_freq, df_sev0, on='IDpol', how='inner')
    df_sev = df_sev[
        ['IDpol', 'Exposure', 'Area', 'VehPower', 'VehAge', 'DrivAge', 'BonusMalus', 'VehBrand', 'VehGas', 'Density',
         'Region', 'ClaimAmount']]

    # Merging severity and frequency datasets -> All dataset
    df_all = pd.merge(df_freq, df_sev0, on='IDpol', how='left')

    # Splitting the set in training dataset and testset
    ids_train, ids_test = train_test_split(df_freq['IDpol'], test_size=0.25, random_state=1)

    freq_train = df_freq.loc[df_freq['IDpol'].isin(ids_train)].set_index('IDpol')
    sev_train = df_sev.loc[df_sev['IDpol'].isin(ids_train)].set_index('IDpol')
    all_train = df_all.loc[df_all['IDpol'].isin(ids_train)].set_index('IDpol')

    freq_test = df_freq.loc[df_freq['IDpol'].isin(ids_test)].set_index('IDpol')
    sev_test = df_sev.loc[df_sev['IDpol'].isin(ids_test)].set_index('IDpol')
    all_test = df_all.loc[df_all['IDpol'].isin(ids_test)].set_index('IDpol')

    return freq_train, sev_train, all_train, freq_test, sev_test, all_test
