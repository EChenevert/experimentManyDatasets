import statistics

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from gplearn.genetic import SymbolicRegressor
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
                                             max_features=4,
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


def dataPredictSR(X_train, y_train, X_test, y_test):
    # est = SymbolicRegressor()
    #     # est.fit(X_train, y_train)
    #     # ypred = est.predict(X_test)
    #     # bestMod = est
    #     # # bestscore = 1
    #     # bestscore = metrics.r2_score(y_test, ypred)
    function_set = ['add', 'sub', 'mul', 'div']

    # Create a based model
    est_gp_grid = SymbolicRegressor(population_size=5000,
                                    n_jobs=-1,
                                    p_crossover=0.6250000000000001,
                                    p_hoist_mutation=0.07499999999999991,
                                    p_point_mutation=0.10833333333333328,
                                    p_subtree_mutation=0.06666666666666658,
                                    parsimony_coefficient=0.001,
                                    const_range=None,
                                    init_method='half and half',
                                    function_set=function_set,
                                    tournament_size=3,  # default value = 20
                                    generations=20,
                                    stopping_criteria=0.01)

    bestMod = est_gp_grid
    bestMod.fit(X_train, y_train)
    ypred = bestMod.predict(X_test)
    bestscore = metrics.r2_score(y_test, ypred)
    return ypred, bestMod, bestscore  # there is no best score becuase there is no grid search


bestscoresls = []
bestmodelsls = []
bestpredictedvalues = []
besttestvalues = []
bestXvals = []
for i in range(1, 101):
    X_train, X_test, y_train, y_test = train_test_split(X[bestfeatures], y, test_size=0.2, shuffle=True)
    ypred, bestmodel, bestscore = dataPredictSR(X_train, y_train, X_test, y_test)
    bestscoresls.append(bestscore)
    bestmodelsls.append(bestmodel)
    bestpredictedvalues.append(ypred)
    besttestvalues.append(y_test)
    bestXvals.append(X_test)

plt.figure()
sns.boxplot(x=bestscoresls, showfliers=False)
plt.show()

# Find the median score for checking purposes
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
axf.set_title('Heatmap 100x Repeated 5-Fold CV for all sites')
axf.text(0.05, 0.95, str('R2: ' + str(round(r2allcv, 4))), transform=axf.transAxes, fontsize=14,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
figf.show()

# Plot best model results
getidx = bestscoresls.index(np.max(bestscoresls))  # GIVES THE INDEX OF THE BEST MODEL!!!!
print(np.max(bestscoresls))  # print the highest score
print(bestmodelsls[getidx]._program)  # Print the coeficients of the best model

# pred_acc = bestmodelsls[getidx].predict(bestXvals[getidx])
r2best = metrics.r2_score(besttestvalues[getidx], bestscoresls[getidx])
fig, ax = plt.subplots()
ax.scatter(besttestvalues[getidx], bestscoresls[getidx])

lims = [
    np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
    np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
]

plt.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
ax.set_aspect('equal')  # can also be equal
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_title('Best Train-Test SR Prediction for Accretion for all sites')
ax.set_xlabel('Observed Accretion Rate (mm/yr)')
ax.set_ylabel('Predicted Accretion Rate (mm/yr)')
textstr = str('R2: ' + str(round(r2best, 4))) + '\n' + str(bestmodelsls[getidx]._program)
ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=8,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
fig.show()

print('################ BEST SPLIT REGRESSION RESULTS###############')
regression_results(besttestvalues[getidx], bestpredictedvalues[getidx])
print(bestfeatures)

