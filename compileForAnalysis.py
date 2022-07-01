import pandas as pd
import numpy as np

# Just using the data created from the everythingCRMS2 repository
fromarcDF = pd.read_csv(r"D:\Etienne\summer2022_CRMS\everythingCRMS2\ALLDATA2.csv", encoding="unicode_escape")\
    .set_index('Simple site')
geedata = pd.read_csv("D:\Etienne\summer2022_CRMS\everythingCRMS2\CRMS_GEE2_redo.csv", encoding="unicode_escape")[
    ['NDVI', 'tss_med', 'windspeed', 'Simple_sit']
].fillna(0).set_index('Simple_sit')
floodfreq = pd.read_csv(r"D:\Etienne\summer2022_CRMS\everythingCRMS2\floodFrequencySitePerYear.csv",
                        encoding="unicode_escape")
geelandarea = pd.read_csv(r"D:\Etienne\summer2022_CRMS\everythingCRMS2\CRMS_GEE_JRCCOPY2.csv", encoding="unicode_escape")[
    ['Simple_sit', 'Basins', 'Community', 'Land_Lost_m2', 'Land_Gained_m2', 'marshType']
].fillna(0).set_index('Simple_sit')

geelandarea['marshType'] = geelandarea['marshType'].replace({'Mineral_Fluvial_Marsh': 'MF',
                                                             'Mineral_Oceanic_Marsh': 'MO',
                                                             'Organic_Fluvial_Marsh': 'OF',
                                                             'Organic_Oceanic_Marsh': 'OO'})

geelandarea['Basins'] = geelandarea['Basins'].replace({'Terrebonne': 'Terre',
                                                        'Barataria': 'Ba',
                                                        'Mermentau': 'Mer',
                                                        'Atchafalaya': 'Atch',
                                                       'Ponchartrain': 'Pon',
                                                       'Brenton Sound': 'BS',
                                                       'Unammed_basin': 'NA',
                                                       'Calcasieu_Sabine': 'CS',
                                                       'Teche_Vermillion': 'TV'})

# Check that all variables / values are correct and representative of enviro
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure()
sns.boxplot(data=geelandarea, x='marshType', y=geelandarea['Land_Gained_m2'] - geelandarea['Land_Lost_m2'],
            showfliers=False)
plt.title('Land Gain minus Land Lost per Marsh Type')
plt.show()  # important to note that not much land lost relative to land gain has occurred

plt.figure()
sns.boxplot(data=geelandarea, x='marshType', y=geelandarea['Land_Gained_m2'], showfliers=False)
plt.title('Land Gain per Marsh Type')
plt.show()

plt.figure()
sns.boxplot(data=geelandarea, x='marshType', y=geelandarea['Land_Lost_m2'], showfliers=False)
plt.title('Land Loss per Marsh Type')
plt.show()

# Land loss / gain by basin
sns.boxplot(data=geelandarea, x='Basins', y=geelandarea['Land_Gained_m2'] - geelandarea['Land_Lost_m2'],
            showfliers=False)
plt.title('Land Gain minus Land Lost per Basin')
plt.show()  # important to note that not much land lost relative to land gain has occurred

plt.figure()
sns.boxplot(data=geelandarea, x='Basins', y=geelandarea['Land_Gained_m2'], showfliers=False)
plt.title('Land Gain per Basin')
plt.show()

plt.figure()
sns.boxplot(data=geelandarea, x='Basins', y=geelandarea['Land_Lost_m2'], showfliers=False)
plt.title('Land Loss per Basin')
plt.show()

# Land loss / gain by Marsh Community
sns.boxplot(data=geelandarea, x='Community', y=geelandarea['Land_Gained_m2'] - geelandarea['Land_Lost_m2'],
            showfliers=False)
plt.title('Land Gain minus Land Lost per Community')
plt.show()  # important to note that not much land lost relative to land gain has occurred

plt.figure()
sns.boxplot(data=geelandarea, x='Community', y=geelandarea['Land_Gained_m2'], showfliers=False)
plt.title('Land Gain per Community')
plt.show()

plt.figure()
sns.boxplot(data=geelandarea, x='Community', y=geelandarea['Land_Lost_m2'], showfliers=False)
plt.title('Land Loss per Community')
plt.show()


# Check if RANSAC and Accretion Rate are similar
plt.figure()
sns.scatterplot(data=fromarcDF, x='Accretion Rate (mm/yr)', y='Accretion Rate mm/yr (slope value)')
plt.title("RANSAC Regression Accretion Rate versus Accretion Rate from Date time")
plt.show()


# Concatenate the datasets for more exploratory analysis
df = pd.concat([fromarcDF.drop(['Community', 'Basins', 'marshType'], axis=1), geedata, geelandarea], axis=1)
#
# plt.figure()
# sns.scatterplot(data=df, x='Accretion Rate (mm/yr)', y=geelandarea['Land_Gained_m2'] - geelandarea['Land_Lost_m2'], hue='marshType')
# plt.title("Accretion Rate mm/yr versus Land Loss m2 with Marsh Class Hue")
# plt.show()
#
# plt.figure()
# sns.scatterplot(data=df, x=np.log(df['Accretion Rate (mm/yr)']), y=np.log(df['Land_Gained_m2']), hue='marshType')
# plt.title("Accretion Rate mm/yr versus Land Gained m2 with Marsh Class Hue")
# plt.show()
#
# plt.figure()
# sns.scatterplot(data=df, x=np.log(df['Accretion Rate (mm/yr)']), y=np.log(geelandarea['Land_Gained_m2'] - geelandarea['Land_Lost_m2']), hue='marshType')
# plt.title("Accretion Rate mm/yr versus Land Loss m2 with Marsh Class Hue")
# plt.show()
#
# plt.figure()
# sns.scatterplot(data=df, x=np.log(df['Accretion Rate (mm/yr)']), y=np.log(df['Land_Gained_m2']), hue='marshType')
# plt.title("Accretion Rate mm/yr versus Land Gained m2 with Marsh Class Hue")
# plt.show()
#
#
#
# plt.figure()
# sns.scatterplot(data=df, x='Accretion Rate (mm/yr)', y='Land_Lost_m2', hue='Basins')
# plt.title("Accretion Rate mm/yr versus Land Loss m2 with Basin Hue")
# plt.show()
#
# plt.figure()
# sns.scatterplot(data=df, x=np.log(df['Accretion Rate (mm/yr)']), y=np.log(df['Land_Gained_m2']), hue='Basins')
# plt.title("Accretion Rate mm/yr versus Land Gained m2 with Basin Hue")
# plt.show()
#
# plt.figure()
# sns.scatterplot(data=df, x=np.log(df['Accretion Rate (mm/yr)']), y=np.log(df['Land_Lost_m2']), hue='Basins')
# plt.title("Accretion Rate mm/yr versus Land Loss m2 with Basin Hue")
# plt.show()
#
# plt.figure()
# sns.scatterplot(data=df, x=np.log(df['Accretion Rate (mm/yr)']), y=np.log(df['Land_Gained_m2']), hue='Basins')
# plt.title("Accretion Rate mm/yr versus Land Gained m2 with Basin Hue")
# plt.show()
#
#
# plt.figure()
# sns.scatterplot(data=df, x='Accretion Rate (mm/yr)', y='Land_Lost_m2', hue='Community')
# plt.title("Accretion Rate mm/yr versus Land Loss m2 with Community Hue")
# plt.show()
#
# plt.figure()
# sns.scatterplot(data=df, x=np.log(df['Accretion Rate (mm/yr)']), y=np.log(df['Land_Gained_m2']), hue='Community')
# plt.title("Accretion Rate mm/yr versus Land Gained m2 with Community Hue")
# plt.show()
#
# plt.figure()
# sns.scatterplot(data=df, x=np.log(df['Accretion Rate (mm/yr)']), y=np.log(df['Land_Lost_m2']), hue='Community')
# plt.title("Accretion Rate mm/yr versus Land Loss m2 with Community Hue")
# plt.show()
#
# plt.figure()
# sns.scatterplot(data=df, x=np.log(df['Accretion Rate (mm/yr)']), y=np.log(df['Land_Gained_m2']), hue='Community')
# plt.title("Accretion Rate mm/yr versus Land Gained m2 with Community Hue")
# plt.show()
#
# # Individual plots of marsh type and accretion v land lost
# plt.figure()
# sns.scatterplot(data=df[df['marshType'] == 'MO'], x=np.log(df[df['marshType'] == 'MO']['Accretion Rate (mm/yr)']),
#                 y=np.log(df[df['marshType'] == 'MO']['Land_Lost_m2']))
# plt.title("Accretion Rate mm/yr versus Land Gained m2 with Mineral Oceanic")
# plt.show()
#
# plt.figure()
# sns.scatterplot(data=df[df['marshType'] == 'MF'], x=np.log(df[df['marshType'] == 'MF']['Accretion Rate (mm/yr)']),
#                 y=np.log(df[df['marshType'] == 'MF']['Land_Lost_m2']))
# plt.title("Accretion Rate mm/yr versus Land Gained m2 with Mineral Fluvial")
# plt.show()
#
#
# # By marsh community
# plt.figure()
# sns.scatterplot(data=df[df['Community'] == 'Saline'], x=np.log(df[df['Community'] == 'Saline']['Accretion Rate (mm/yr)']),
#                 y=np.log(df[df['Community'] == 'Saline']['Land_Lost_m2']))
# plt.title("Accretion Rate mm/yr versus Land Gained m2 with Saline")
# plt.show()
#
# plt.figure()
# sns.scatterplot(data=df[df['Community'] == 'Freshwater'], x=np.log(df[df['Community'] == 'Freshwater']['Accretion Rate (mm/yr)']),
#                 y=np.log(df[df['Community'] == 'Freshwater']['Land_Lost_m2']))
# plt.title("Accretion Rate mm/yr versus Land Gained m2 with Freshwater")
# plt.show()
#
# # Fluvial and Oceanic
# odf = df[(df['marshType'] == 'MO') | (df['marshType'] == 'OO')]
# plt.figure()
# sns.scatterplot(data=odf, x=np.log(odf['Accretion Rate (mm/yr)']), y=np.log(odf['Land_Lost_m2']))
# plt.title("Accretion Rate mm/yr versus Land Lost m2 with Oceanic")
# plt.show()
#
# fdf = df[(df['marshType'] == 'MF') | (df['marshType'] == 'OF')]
# plt.figure()
# sns.scatterplot(data=odf, x=np.log(fdf['Accretion Rate (mm/yr)']), y=np.log(fdf['Land_Lost_m2']))
# plt.title("Accretion Rate mm/yr versus Land Lost m2 with Fluvial")
# plt.show()
#
# # brackish/saline and freshwater/swamp
# odf = df[(df['Community'] == 'Saline') | (df['Community'] == 'Brackish')]
# plt.figure()
# sns.scatterplot(data=odf, x=odf['Accretion Rate (mm/yr)'], y=np.log(odf['Land_Lost_m2']))
# plt.title("Accretion Rate mm/yr versus Land Lost m2 with Saline/Brackish")
# plt.show()
#
# fdf = df[(df['Community'] == 'Freshwater') | (df['Community'] == 'Swamp')]
# plt.figure()
# sns.scatterplot(data=odf, x=fdf['Accretion Rate (mm/yr)'], y=np.log(fdf['Land_Lost_m2']))
# plt.title("Accretion Rate mm/yr versus Land Lost m2 with Freshwater/Swamp")
# plt.show()

# checking if twilleys recordings are in crms
tdf = df[(df['Basins'] == 'Terre')]
plt.figure()
sns.scatterplot(data=tdf, x=np.log(tdf['Accretion Rate (mm/yr)']), y=np.log(tdf['Land_Lost_m2']))
plt.title("Accretion Rate mm/yr versus Land Lost m2 in Terrebonne")
plt.show()

amdf = df[(df['Basins'] == 'Atch') | (df['Basins'] == 'BS')]
plt.figure()
sns.scatterplot(data=amdf, x=np.log(amdf['Accretion Rate (mm/yr)']), y=np.log(amdf['Land_Lost_m2']))
plt.title("Accretion Rate mm/yr versus Land Lost m2 in Atchafalaya and BS")
plt.show()

# mineral oceanic and organic fluvial
modf = df[(df['marshType'] == 'MO')]
plt.figure()
sns.scatterplot(data=modf, x=np.log(modf['Accretion Rate (mm/yr)']), y=np.log(modf['Land_Lost_m2']))
plt.title("Accretion Rate mm/yr versus Land Lost m2 in Mineral oceanic")
plt.show()

ofdf = df[(df['marshType'] == 'MF')]
plt.figure()
sns.scatterplot(data=amdf, x=np.log(ofdf['Accretion Rate (mm/yr)']), y=np.log(ofdf['Land_Lost_m2']))
plt.title("Accretion Rate mm/yr versus Land Lost m2 in Mineral Fluvial")
plt.show()

#
# fromarcDF = fromarcDF.rename(columns={'POLY_AREA': 'Nearest Lake Area m2', 'PERIMETER': 'Nearest Lake Perimeter m'})
# df = pd.concat([fromarcDF, geedata], axis=1)
#
# ## Add other External Variables from CRMS
# ## Percent time flooded, land:water ratio, fetch ??, tide amp/range, ...
# # Add percent flooded
#
# Attach variables
def hydro_groupby(filename):
    """

    :param filename:
    :return:
    """
    perc = pd.read_csv(filename, encoding='unicode escape')
    perc_i = perc[[
        'Station_ID', 'avg_percentflooded (%)', 'percent_waterlevel_complete'  #'Calendar_Year','min_date',
    ]]
    perc_i['Simple site'] = [i[:8] for i in perc_i['Station_ID']]
    perc_gb = perc_i.groupby(['Simple site']).median()  # excludes the outliers
    return perc_gb


# Add 2016 to 2022
# set_index_df = add_basins_df.set_index(['Simple site', 'Year (yyyy)', 'Season'])
flood_2016_2022 = hydro_groupby(r"D:\Etienne\summer2022_CRMS\percentFlooded\percentFlooded_2016_2022.csv")
flood_2011_2016 = hydro_groupby(r"D:\Etienne\summer2022_CRMS\percentFlooded\percentFlooded_2011_2016.csv")
flood_2006_2011 = hydro_groupby(r"D:\Etienne\summer2022_CRMS\percentFlooded\percentFlooded_2006_2011.csv")

concat_test = pd.concat([flood_2006_2011, flood_2011_2016, flood_2016_2022], join='outer', axis=1)
gb_m = concat_test.groupby(concat_test.columns, axis=1).median()
add_flood_df = pd.concat([df, gb_m.drop('percent_waterlevel_complete', axis=1)], join='inner', axis=1)

# Add Tides
def waterlevel_groupby(filename):
    """

    :param filename:
    :return:
    """
    perc = pd.read_csv(filename, encoding='unicode escape')
    perc_i = perc[[
        'Station_ID', 'Tide_Amp (ft)', 'avg_water_level (ft NAVD88)', 'avg_flooding (ft)'
    ]]
    perc_i['Simple site'] = [i[:8] for i in perc_i['Station_ID']]
    perc_gb = perc_i.groupby(['Simple site']).median()  # excludes the outliers
    return perc_gb

wl2006_2022 = waterlevel_groupby(r"D:\Etienne\summer2022_CRMS\waterLevelRangeCRMS\waterLevelcrms2006_2022.csv")

add_wl_df = pd.concat([add_flood_df, wl2006_2022], join='inner', axis=1)

# # Add land fragmentation variable (land percent within a 1km radius) datasets
# def landarea_groupby(filename):
#     """
#
#     :param filename:
#     :return:
#     """
#     perc = pd.read_csv(filename, encoding='unicode escape')
#     perc_i = perc[[
#         'SiteId', 'Percent_Land'
#     ]]
#     perc_i['Simple_sit'] = [i[:8] for i in perc_i['SiteId']]
#     perc_gb = perc_i.groupby(['SiteId']).median()  # excludes the outliers
#     return perc_gb
#
# addlandperc = landarea_groupby("D:\Etienne\summer2022_CRMS\LandArea\landAREA.csv")
# alldata = pd.concat([add_wl_df, addlandperc], join='inner', axis=1)

def addFloodFreq(filename):
    perc = pd.read_csv(filename, encoding='unicode escape')
    perc_i = perc[[
        'Simple site', 'Flood Freq (Floods/yr)'
    ]]
    # perc_i['Simple_sit'] = [i[:8] for i in perc_i['SiteId']]
    perc_gb = perc_i.groupby(['Simple site']).median()  # excludes the outliers
    return perc_gb

addfloodfreq = addFloodFreq("D:\\Etienne\\summer2022_CRMS\\everythingCRMS2\\experimentManyDatasets\\floodFrequencySitePerYear.csv")
alldata = pd.concat([add_wl_df, addfloodfreq], join='inner', axis=1)
# # Drop dumb variables
# final = alldata.drop(
#     [
#         'OBJECTID', 'Join_Count', 'TARGET_FID', 'Join_Count_1', 'TARGET_FID_1', 'Latitude_1', 'Longitude_1'
#     ], axis=1
# )
alldata.to_csv("D:\Etienne\summer2022_CRMS\everythingCRMS2\experimentManyDatasets\CRMS_readyforanalysis7_1.csv")