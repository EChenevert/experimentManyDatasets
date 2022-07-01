import glob
import numpy as np
import pandas as pd

floodingdf = pd.read_csv("D:\Etienne\crmsDATATables\CRMS_Continuous_Hydrographic_2017_04_06_to_present\CRMS_Continuous_Hydrographic_2017_04_06_to_present_HYDROGRAPHIC_HOURLY.csv",
                  encoding="unicode_escape")[['Station ID', 'Adjusted Water Elevation to Marsh (ft)',
                                              'Date (mm/dd/yyyy)']]

# Make the simple site site name
floodingdf['Simple site'] = [i[:8] for i in floodingdf['Station ID']]
floodingdf.drop('Station ID', axis=1)
# divide by time of recording
# floodingdf['date in datetime'] = pd.to_datetime(floodingdf['Date (mm/dd/yyyy)'], format='%m/%d/%Y')
# floodingdf.drop('Date (mm/dd/yyyy)', axis=1)


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


split_bysite_ls = split_by_intrinsic_var(floodingdf, "Simple site")
save_to_folder("D:\\Etienne\\summer2022_CRMS\\everythingCRMS2\\siteFloodFreq", split_bysite_ls, "Simple site")

############## Load all those datasets back in

path = "D:\\Etienne\\summer2022_CRMS\\everythingCRMS2\\siteFloodFreq"

# csv files in the path
files = glob.glob(path + "/*.csv")
print(files)

# checking all the csv files in the
# specified path
listdfs = []
for filename in files:
    # reading content of csv file
    # content.append(filename)
    df = pd.read_csv(filename, encoding="unicode_escape")
    # df = df.drop('Unnamed: 0.1', axis=1)
    # df = df.rename(columns={'Unnamed: 0': 'Simple site'})
    listdfs.append(df)

dictdf = {}
arraydf = {}
for df in listdfs:
    dfname = df['Simple site'][0]  # name of marsh type used to name dataframe and associated features
    dictdf[dfname] = df.dropna().reset_index()

    wlarray = dictdf[dfname]['Adjusted Water Elevation to Marsh (ft)'].to_numpy()
    dictdf[dfname]['date in datetime'] = pd.to_datetime(dictdf[dfname]['Date (mm/dd/yyyy)'], format='%m/%d/%Y')
    start = dictdf[dfname]['date in datetime'][0]
    end = dictdf[dfname]['date in datetime'][len(dictdf[dfname])-1]
    timedays = end - start
    timedays = timedays.days
    decimalyears = timedays / 365
    arraydf[dfname] = np.linspace(1, len(wlarray), len(wlarray))
    for i in range(len(wlarray)):
        if (wlarray[i] <= 0 and wlarray[i-1] > 0) or (wlarray[i] > 0 and wlarray[i-1] <= 0):
            if decimalyears > 0:
                arraydf[dfname][i] = 0.5/decimalyears  # 0.5 cuz I will be appending every time a flood comes in and then out (so i divide)
            else:
                arraydf[dfname][i] = 0.5
        else:
            arraydf[dfname][i] = 0.0

npcolstack = {}
for key in arraydf:
    npcolstack[key] = np.column_stack((dictdf[key]['Simple site'].to_numpy(), arraydf[key]))

stackeddf = {}
for key in npcolstack:
    stackeddf[key] = pd.DataFrame(npcolstack[key], columns=['Simple site', 'Flood Freq (Floods/yr)'])\
        .groupby('Simple site').sum().reset_index()


check = pd.concat(stackeddf.values(), ignore_index=True)
check.to_csv("D:\\Etienne\\summer2022_CRMS\\everythingCRMS2\\experimentManyDatasets\\floodFrequencySitePerYear.csv")

