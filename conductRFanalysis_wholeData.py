import statistics

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import shap
from mlxtend.feature_selection import ExhaustiveFeatureSelector
from numpy import mean
from scipy import stats

# Analysis on whole dataset
from scipy.stats import sem
from sklearn import metrics, linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, GridSearchCV, RepeatedKFold, cross_val_score, train_test_split, \
    cross_val_predict
from sklearn.preprocessing import MinMaxScaler

d = pd.read_csv(r"D:\Etienne\summer2022_CRMS\everythingCRMS2\experimentManyDatasets\CRMS_readyforanalysis7_1.csv",
                encoding="unicode_escape")

outcome = 'Accretion Rate (mm/yr)'  # Define the target variable

allfeats = [
    'Distance_to_Bays_m', 'Distance_to_Ocean_m', 'Distance_to_Fluvial_m',
    'NDVI', 'tss_med', 'windspeed', 'Land_Lost_m2',
    'Soil Salinity (ppt)', 'Average Height Herb (cm)',
    'avg_percentflooded (%)', 'Tide_Amp (ft)', 'avg_flooding (ft)', 'Flood Freq (Floods/yr)',
    outcome
]

# allfeats = [
#     'Distance_to_Ocean_m', 'Distance_to_Fluvial_m', 'Land_Lost_m2', 'Tide_Amp (ft)', 'avg_flooding (ft)',
#     'Flood Freq (Floods/yr)',
#     outcome
# ]

d = d[allfeats].dropna()

# plt.figure()
# sns.pairplot(d)
# plt.show()

# d = d.apply(pd.to_numeric)
for col in d.columns.values:
    d[col + "_z"] = stats.zscore(
        d[col])  # Compute the zscore for each value per column ; save as "col"_z
length = len(allfeats)
for col in d.columns.values[
           length:]:  # 13 to 14 because I added the K_mean_marshType col for visualuization

    d = d[np.abs(d[col]) < 2]  # keep if value is less than 2 std

d = d.drop(  # drop zscore columns
    d.columns.values[length:], axis=1)

# sns.pairplot(d)
# plt.show()

# transformations
d['Distance_to_Bays_m_log'] = [np.log(i) if i > 0 else 0 for i in d['Distance_to_Bays_m']]
d['Distance_to_Fluvial_m_log'] = [np.log(i) if i > 0 else 0 for i in d['Distance_to_Fluvial_m']]
d['Distance_to_Ocean_m_log'] = [np.log(i) if i > 0 else 0 for i in d['Distance_to_Ocean_m']]
d['Land_Lost_m2_log'] = [np.log(i) if i > 0 else 0 for i in d['Land_Lost_m2']]
# After review i drop the less fit variables
d = d.drop(['Distance_to_Bays_m', 'Distance_to_Fluvial_m', 'Land_Lost_m2_log', 'Distance_to_Ocean_m'], axis=1)
# Scale variables
y = d[outcome]
X = d.drop(outcome, axis=1)
scaler = MinMaxScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns.values)

# # Check residuals from cross validation to assess bias in model
# for var in X.columns.values:
#     predictedcvRepeat = []
#     holdtrue = []
#     holdmodels = []
#     holdX = []
#
#     x = np.asarray(X[var]).reshape(-1, 1)
#     splits = KFold(n_splits=5).split(x, y)
#     model = linear_model.LinearRegression()
#     predicted = cross_val_predict(model, x, y, cv=splits)  # RepeatedKFold(n_splits=5, n_repeats=100, random_state=1)
#     predictedcvRepeat.append(predicted)
#     holdtrue.append(y)
#     holdmodels.append(model)
#     holdX.append(np.asarray(x))
#     # Flatten the lists
#     cvpred = [xs for xs in predictedcvRepeat]
#     cvtrue = [xs for xs in holdtrue]
#     cvX = np.squeeze(np.asarray([xs for xs in holdX]))
#
#     # make a residual plot
#     residuals = np.squeeze(np.asarray(cvtrue) - np.asarray(cvpred))
#     plt.figure()
#     sns.scatterplot(x=cvX, y=residuals)
#     plt.title("residual plot of " + str(var))
#     plt.show()


# split data into test train and validation sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
X_train2, X_val, y_train2, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)
# using flow from: https://scikit-learn.org/stable/modules/cross_validation.html
# Exhaustive Feature Selection for MLR
lr = linear_model.LinearRegression()
feature_selector = ExhaustiveFeatureSelector(lr,
                                             min_features=1,
                                             max_features=10,
                                             scoring='r2',
                                             # print_progress=True,
                                             cv=5)  # 5 fold cross-validation
efsmlr = feature_selector.fit(X_train2, y_train2)
# bestfeatures[key] = feature_selector

print('Best CV r2 score: %.2f' % efsmlr.best_score_)
print('Best subset (indices):', efsmlr.best_idx_)
print('Best subset (corresponding names):', efsmlr.best_feature_names_)

# Build a model from identified features
dfmlr = pd.DataFrame.from_dict(efsmlr.get_metric_dict()).T
dfmlr.sort_values('avg_score', inplace=True, ascending=False)

# save best features
bestfeatures = list(efsmlr.best_feature_names_)


# Retrain model wth best parameters
# Make regression with best subset: Selected based on whole dataset
# Re-split data many times and train model. Make 2D histogram and best test v pred scatter plot

def regression_results(y_true2, y_pred2):
    # Regression metrics
    explained_variance = metrics.explained_variance_score(y_true2, y_pred2)
    mean_absolute_error = metrics.mean_absolute_error(y_true2, y_pred2)
    r2 = metrics.r2_score(y_true2, y_pred2)
    mse = metrics.mean_squared_error(y_true2, y_pred2)
    median_absolute_error = metrics.median_absolute_error(y_true2, y_pred2)

    print('explained_variance: ', round(explained_variance, 4))
    print('MAE: ', round(mean_absolute_error, 4))
    print('MSE: ', round(mse, 4))
    print('RMSE: ', round(np.sqrt(mse), 4))
    print('R2: ', round(r2, 4))


def dataPredictRF(X_train, y_train, X_test, y_test):
    model = RandomForestRegressor(n_estimators=400, max_depth=90, min_samples_split=10, min_samples_leaf=4,
                                  bootstrap=True, max_features=len(X_train.columns.values)-1)
    model.fit(X_train, y_train)
    ypred = model.predict(X_test)
    bestMod = model
    # bestscore = 1
    bestscore = metrics.r2_score(y_test, ypred)
    return ypred, bestMod, bestscore  # there is no best score becuase there is no grid search


bestscoresls = []
bestmodelsls = []
bestpredictedvalues = []
besttestvalues = []
bestXvals = []
bestXtrainvals = []
for i in range(1, 101):
    X_train, X_test, y_train, y_test = train_test_split(X[bestfeatures], y, test_size=0.2, shuffle=True)
    ypred, bestmodel, bestscore = dataPredictRF(X_train, y_train, X_test, y_test)
    bestscoresls.append(bestscore)
    bestmodelsls.append(bestmodel)
    bestpredictedvalues.append(ypred)
    besttestvalues.append(y_test)
    bestXvals.append(X_test)
    bestXtrainvals.append(X_train)

plt.figure()
sns.boxplot(x=bestscoresls)
plt.show()

# Grab the index before the sorting of bestscoresls
getidx = bestscoresls.index(np.max(bestscoresls))  # GIVES THE INDEX OF THE BEST MODEL!!!!
print(np.max(bestscoresls))  # print the highest score

bestscoresls.sort()
middlelist = round((len(bestscoresls)-1)/2)
medianScore = bestscoresls[middlelist]
# Print the regression results
besttestvalues = np.asarray(besttestvalues)
allpredicted = []
alltestvals = []
for i in range(len(bestpredictedvalues)):
    for j in range(len(bestpredictedvalues[i])):
        allpredicted.append(bestpredictedvalues[i][j])
        alltestvals.append(besttestvalues[i][j])

# bestpredictedvalues = [x for xs in bestpredictedvalues for x in xs]
# besttestvalues = [x for xs in besttestvalues for x in xs]

regression_results(alltestvals, allpredicted)
r2allcv = metrics.r2_score(alltestvals, allpredicted)
figf, axf = plt.subplots()
lims = [
    np.min([alltestvals, allpredicted]),  # min of both axes
    np.max([alltestvals, allpredicted]),  # max of both axes
]
axf.plot(lims, lims, marker="o", alpha=0.5)
axf.hist2d(alltestvals, allpredicted, bins=50, cmap='YlGn')

axf.set_aspect('equal')  # can also be equal
axf.set_xlabel('Observed Accretion Rate (mm/yr)')
axf.set_ylabel('Predicted Accretion Rate (mm/yr)')
axf.set_title('RF Heatmap 100x Repeated 5-Fold CV for all sites')
axf.text(0.05, 0.95, str('R2: ' + str(round(r2allcv, 4))), transform=axf.transAxes, fontsize=14,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
figf.show()

# Plot best model results

# print(bestmodelsls[getidx].coef_)  # Print the coeficients of the best model

# pred_acc = bestmodelsls[getidx].predict(bestXvals[getidx])
r2best = metrics.r2_score(besttestvalues[getidx], bestpredictedvalues[getidx])
fig, ax = plt.subplots()
ax.scatter(besttestvalues[getidx], bestpredictedvalues[getidx])

lims = [
    np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
    np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
]

plt.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
ax.set_aspect('equal')  # can also be equal
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_title('Best Train-Test RF Prediction for Accretion for all sites')
ax.set_xlabel('Observed Accretion Rate (mm/yr)')
ax.set_ylabel('Predicted Accretion Rate (mm/yr)')
textstr = str('R2: ' + str(round(r2best, 4)))
ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
fig.show()

print('################ BEST SPLIT REGRESSION RESULTS###############')
regression_results(besttestvalues[getidx], bestpredictedvalues[getidx])
print(bestfeatures)


# add SHAPLEY
bestmodel = bestmodelsls[getidx]
data = bestXtrainvals[getidx]

shap_values = shap.TreeExplainer(bestmodel).shap_values(data)


def ABS_SHAP(df_shap, df):
    """
    from -> https://medium.com/dataman-in-ai/explain-your-model-with-the-shap-values-bc36aac4de3d
    :param df_shap:
    :param df:
    :return:
    """
    # import matplotlib as plt
    # Make a copy of the input data
    shap_v = pd.DataFrame(df_shap)
    feature_list = df.columns
    shap_v.columns = feature_list
    df_v = df.copy().reset_index().drop('index', axis=1)

    # Determine the correlation in order to plot with different colors
    corr_list = list()
    for i in feature_list:
        b = np.corrcoef(shap_v[i], df_v[i])[1][0]
        corr_list.append(b)
    corr_df = pd.concat([pd.Series(feature_list), pd.Series(corr_list)], axis=1).fillna(0)
    # Make a data frame. Column 1 is the feature, and Column 2 is the correlation coefficient
    corr_df.columns = ['Variable', 'Corr']
    corr_df['Sign'] = np.where(corr_df['Corr'] > 0, 'red', 'blue')

    # Plot it
    shap_abs = np.abs(shap_v)
    k = pd.DataFrame(shap_abs.mean()).reset_index()
    k.columns = ['Variable', 'SHAP_abs']
    k2 = k.merge(corr_df, left_on='Variable', right_on='Variable', how='inner')
    k2 = k2.sort_values(by='SHAP_abs', ascending=True)
    colorlist = k2['Sign']

    ax = k2.plot.barh(x='Variable', y='SHAP_abs', color=colorlist, figsize=(15, 8), legend=False)
    ax.set_xlabel("SHAP Value (Red = Positive Impact)")
    ax.set_title("Feature Importances of Training Dataset used for Best prediction")
    figure = ax.get_figure()

    return figure


figure = ABS_SHAP(shap_values, data)
figure.show()
# figure.savefig(featureimpPath + "wholeDataset.png")

