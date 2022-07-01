import numpy as np
import pandas as pd

def split_by_intrinsic_var(df, var_str):
    unq_vals = df[var_str].unique().tolist()
    hold_splits = []
    for value in unq_vals:
        hold_splits.append(df.groupby(var_str).get_group(value).reset_index().drop('index', axis=1))
    return hold_splits

def save_to_folder(folder_path, df_list, group_col):
    """

    :param folder_path: path to a folder. Must have \\ path separation if in windows
    :param df_list: list of grouped dataframes
    :return: None, Exports files to place on computer
    """
    folder_path = folder_path
    for df in df_list:
        group_str = df[group_col][0]
        df.to_csv(folder_path+"\\"+group_str+".csv")

d = pd.read_csv(r"D:\Etienne\summer2022_CRMS\everythingCRMS2\experimentManyDatasets\CRMS_readyforanalysis7_1.csv", encoding="unicode_escape").dropna(subset='Total Mass Accumulation (g/yr)')
d = d.rename(columns={'windspeed': 'windspeed (m/s)'})

split_bysite_ls = split_by_intrinsic_var(d, "marshType")
save_to_folder("D:\\Etienne\\summer2022_CRMS\\everythingCRMS2\\experimentManyDatasets\\marshTypeDatasets", split_bysite_ls, "marshType")

##### Concatenate into Biotic, Fluvial, and Oceanic datasets
fluvial = pd.concat([split_bysite_ls[0].set_index('Unnamed: 0'), split_bysite_ls[2].set_index('Unnamed: 0')], axis=0,
                    join='outer')
fluvial['marshControl'] = np.asarray(
    ['Fluvial' for i in range(len(fluvial))]
)
oceanic = pd.concat([split_bysite_ls[1].set_index('Unnamed: 0'), split_bysite_ls[3].set_index('Unnamed: 0')], axis=0,
                    join='outer')
oceanic['marshControl'] = np.asarray(
    ['oceanic' for i in range(len(oceanic))]
)
biotic = pd.concat([split_bysite_ls[2].set_index('Unnamed: 0'), split_bysite_ls[3].set_index('Unnamed: 0')], axis=0,
                   join='outer')
biotic['marshControl'] = np.asarray(
    ['biotic' for i in range(len(biotic))]
)
mineral = pd.concat([split_bysite_ls[0].set_index('Unnamed: 0'), split_bysite_ls[1].set_index('Unnamed: 0')], axis=0,
                    join='outer')
mineral['marshControl'] = np.asarray(
    ['mineral' for i in range(len(mineral))]
)
fluvial.to_csv("D:\\Etienne\\summer2022_CRMS\\everythingCRMS2\\experimentManyDatasets\\marshTypeDatasets\\accControlDatasets\\fluvial.csv")
oceanic.to_csv("D:\\Etienne\\summer2022_CRMS\\everythingCRMS2\\experimentManyDatasets\\marshTypeDatasets\\accControlDatasets\\oceanic.csv")
biotic.to_csv("D:\\Etienne\\summer2022_CRMS\\everythingCRMS2\\experimentManyDatasets\\marshTypeDatasets\\accControlDatasets\\biotic.csv")
mineral.to_csv("D:\\Etienne\\summer2022_CRMS\\everythingCRMS2\\experimentManyDatasets\\marshTypeDatasets\\accControlDatasets\\mineral.csv")



