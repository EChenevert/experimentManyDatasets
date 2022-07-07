import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
from sklearn import metrics


df = pd.read_csv(r"D:\Etienne\summer2022_CRMS\everythingCRMS2\experimentManyDatasets\CRMS_readyforanalysis7_1.csv",
                 encoding="unicode_escape").dropna(subset='Total Mass Accumulation (g/yr)')

# Investigating the relation between land loss and accretion rate in saline marshes, terrebonne, amd oceanic marshes
# Saline and brackish marshs

# # Log transform
# df['Land_Lost_m2_log'] = [np.log(i) if i > 0 else 0 for i in df['Land_Lost_m2']]

sb = df[(df['Community'] == 'Saline')]
sns.regplot(data=sb, x='Land_Lost_m2', y='Accretion Rate (mm/yr)')
plt.title('Saline Marsh Communities')
plt.show()
# Fit sm.ols regression and print model summary
acc_model = sm.OLS(sb['Accretion Rate (mm/yr)'], sb['Land_Lost_m2']).fit()
print(acc_model.summary())
mapeSaline = metrics.mean_absolute_percentage_error(sb['Accretion Rate (mm/yr)'], acc_model.predict(sb['Land_Lost_m2']))

# Investigating Freshwater Marshes
sb = df[(df['Community'] == 'Freshwater')]
sns.regplot(data=sb, x='Land_Lost_m2', y='Accretion Rate (mm/yr)')
plt.title('Freshwater Marsh Communities')
plt.show()
# Fit sm.ols regression and print model summary
acc_model = sm.OLS(sb['Accretion Rate (mm/yr)'], sb['Land_Lost_m2']).fit()
print(acc_model.summary())
mapeFreshwater = metrics.mean_absolute_percentage_error(sb['Accretion Rate (mm/yr)'], acc_model.predict(sb['Land_Lost_m2']))

# Investigating Terrebonne Basin
sb = df[(df['Basins'] == 'Terre')]
sns.regplot(data=sb, x='Land_Lost_m2', y='Accretion Rate (mm/yr)')
plt.title('Terrebonne Hydro Basin')
plt.show()
# Fit sm.ols regression and print model summary
acc_model = sm.OLS(sb['Accretion Rate (mm/yr)'], sb['Land_Lost_m2']).fit()
print(acc_model.summary())
mapeTerre = metrics.mean_absolute_percentage_error(sb['Accretion Rate (mm/yr)'], acc_model.predict(sb['Land_Lost_m2']))

# Investigate Barataria Basin
sb = df[(df['Basins'] == 'Ba')]
sns.regplot(data=sb, x='Land_Lost_m2', y='Accretion Rate (mm/yr)')
plt.title('Barataria Hydro Basin')
plt.show()
# Fit sm.ols regression and print model summary
acc_model = sm.OLS(sb['Accretion Rate (mm/yr)'], sb['Land_Lost_m2']).fit()
print(acc_model.summary())
mapeBara = metrics.mean_absolute_percentage_error(sb['Accretion Rate (mm/yr)'], acc_model.predict(sb['Land_Lost_m2']))


# Investigating the Oceanic Marshes
sb = df[(df['marshType'] == 'MO') | (df['marshType'] == 'OO')]
sns.regplot(data=sb, x='Land_Lost_m2', y='Accretion Rate (mm/yr)')
plt.title('Oceanic Marshes')
plt.show()
# Fit sm.ols regression and print model summary
acc_model = sm.OLS(sb['Accretion Rate (mm/yr)'], sb['Land_Lost_m2']).fit()
print(acc_model.summary())
mapeOcean = metrics.mean_absolute_percentage_error(sb['Accretion Rate (mm/yr)'], acc_model.predict(sb['Land_Lost_m2']))

# Investigating Oceanic Mineral Marshes
sb = df[(df['marshType'] == 'MO')]
sns.regplot(data=sb, x='Land_Lost_m2', y='Accretion Rate (mm/yr)')
plt.title('Mineral Oceanic Marshes')
plt.show()
# Fit sm.ols regression and print model summary
acc_model = sm.OLS(sb['Accretion Rate (mm/yr)'], sb['Land_Lost_m2']).fit()
print(acc_model.summary())
mapeMO = metrics.mean_absolute_percentage_error(sb['Accretion Rate (mm/yr)'], acc_model.predict(sb['Land_Lost_m2']))


# Investigating Oceanic Mineral Marshes
sb = df[(df['marshType'] == 'OO')]
sns.regplot(data=sb, x='Land_Lost_m2', y='Accretion Rate (mm/yr)')
plt.title('Organic Oceanic Marshes')
plt.show()
# Fit sm.ols regression and print model summary
acc_model = sm.OLS(sb['Accretion Rate (mm/yr)'], sb['Land_Lost_m2']).fit()
print(acc_model.summary())
mapeOO = metrics.mean_absolute_percentage_error(sb['Accretion Rate (mm/yr)'], acc_model.predict(sb['Land_Lost_m2']))

# Investigating Fluvial marshes
sb = df[(df['marshType'] == 'OF') | (df['marshType'] == 'MF')]
sns.regplot(data=sb, x='Land_Lost_m2', y='Accretion Rate (mm/yr)')
plt.title('Fluvial Marshes')
plt.show()
# Fit sm.ols regression and print model summary
acc_model = sm.OLS(sb['Accretion Rate (mm/yr)'], sb['Land_Lost_m2']).fit()
print(acc_model.summary())
mapeFluvial = metrics.mean_absolute_percentage_error(sb['Accretion Rate (mm/yr)'], acc_model.predict(sb['Land_Lost_m2']))

# Investigating Fluvial marshes
sb = df[(df['marshType'] == 'MF')]
sns.regplot(data=sb, x='Land_Lost_m2', y='Accretion Rate (mm/yr)')
plt.title('Mineral Fluvial Marshes')
plt.show()
# Fit sm.ols regression and print model summary
acc_model = sm.OLS(sb['Accretion Rate (mm/yr)'], sb['Land_Lost_m2']).fit()
print(acc_model.summary())
mapeMF = metrics.mean_absolute_percentage_error(sb['Accretion Rate (mm/yr)'], acc_model.predict(sb['Land_Lost_m2']))


# Investigating Fluvial marshes
sb = df[(df['marshType'] == 'OF')]
sns.regplot(data=sb, x='Land_Lost_m2', y='Accretion Rate (mm/yr)')
plt.title('Organic Fluvial Marshes')
plt.show()
# Fit sm.ols regression and print model summary
acc_model = sm.OLS(sb['Accretion Rate (mm/yr)'], sb['Land_Lost_m2']).fit()
print(acc_model.summary())
mapeOF = metrics.mean_absolute_percentage_error(sb['Accretion Rate (mm/yr)'], acc_model.predict(sb['Land_Lost_m2']))


