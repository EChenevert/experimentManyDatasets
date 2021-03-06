import statistics
import shap
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
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

d = d[allfeats].dropna()  # extract the most important features identified for that dataframe

########## Paths to hold plot results #####################

featureimpPath = "D:\\Etienne\\summer2022_CRMS\\everythingCRMS2\\experimentManyDatasets\\FeatureImportancePlots\\"
MLRtesttrainPath = "D:\\Etienne\\summer2022_CRMS\\everythingCRMS2\\experimentManyDatasets\\MLRtest_trainPlots\\"
RFtesttrainPlot = "D:\\Etienne\\summer2022_CRMS\\everythingCRMS2\\experimentManyDatasets\\RFtest_trainPlots\\"
RF2DtesttrainPlot = "D:\\Etienne\\summer2022_CRMS\\everythingCRMS2\\experimentManyDatasets\\RF2Dtest_trainPlots\\"

######### --------------------------- #######################

# dictdf[dfname][outcome] = [np.exp(i) if i != 0 else 0 for i in dictdf[dfname][outcome]]  # because the mineral mass acc was logged prior to loading in

# d = d.apply(pd.to_numeric)
for col in d.columns.values:
    d[col + "_z"] = stats.zscore(
        d[col])  # Compute the zscore for each value per column ; save as "col"_z
length = len(allfeats)
for col in d.columns.values[
           length:]:  # 13 to 14 because I added the K_mean_marshType col for visualuization

    d = d[np.abs(d[col]) < 3]  # keep if value is less than 2 std

d = d.drop(  # drop zscore columns
    d.columns.values[length:], axis=1)

# sns.pairplot(d)
# plt.show()

howtoscore = 'r2'


def get_hyperRF(folds):
    model = RandomForestRegressor(n_estimators=400, max_depth=90, random_state=0)
    paramgrid = {'min_samples_split': [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20]}
    gridsearch = GridSearchCV(estimator=model, param_grid=paramgrid, cv=folds, n_jobs=-1, verbose=3,
                              scoring=howtoscore)
    return gridsearch


def get_rf():
    model = RandomForestRegressor(n_estimators=400, max_depth=90, random_state=0)
    return model


def evaluate_modelrf(data, target, folds=10, repeats=100):
    # prepare the cross-validation procedure
    cv = RepeatedKFold(n_splits=folds, n_repeats=repeats, random_state=1)
    model = get_rf()
    model.fit(data, y)
    # model = Lasso()
    # evaluate model
    scores = cross_val_score(model, data, target, scoring=howtoscore, cv=cv, n_jobs=-1)

    return scores


def evaluate_modelhRF(data, target, folds=10, repeats=100):
    # prepare the cross-validation procedure
    cv = RepeatedKFold(n_splits=folds, n_repeats=repeats, random_state=1)
    model = get_hyperRF(folds)
    model.fit(data, y)
    model = model.best_estimator_
    # model = Lasso()
    # evaluate model
    scores = cross_val_score(model, data, target, scoring=howtoscore, cv=cv, n_jobs=-1)

    return scores


def averagingfunc(ls):
    """
    This way it is easier to change between the median of the 100x CV scores and mean of the 100x CV scores
    :param ls:
    :return:
    """
    means = mean(ls)
    return means


# Scale
# Standardizing
y = d[outcome]
X = d.drop(outcome, axis=1)
scaler = MinMaxScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns.values)

results = list()
# evaluate using a given number of repeats
scores = evaluate_modelrf(X, y, 5, 100)
# summarize
print('>%d avg_score=%.4f se=%.3f' % (100, averagingfunc(scores), sem(scores)))
scoresdict = scores
meanscores = averagingfunc(scores)
maxscores = np.max(scores)
semscores = sem(scores)
results.append(scores)

# Cross validation plot
# predicted = cross_val_predict(get_hyperRF(5).fit(X, y).best_estimator_, X, y,
#                               cv=5)  # RepeatedKFold(n_splits=5, n_repeats=100, random_state=1)
predictedcvRepeat = []
holdtrue = []
holdmodels = []
for r in range(1, 251):
    splits = KFold(n_splits=5, shuffle=True).split(X, y)
    model = get_rf().fit(X, y)
    predicted = cross_val_predict(model, X, y,
                                  cv=splits)  # RepeatedKFold(n_splits=5, n_repeats=100, random_state=1)
    predictedcvRepeat.append(predicted)
    holdtrue.append(y)
    holdmodels.append(model)
# flatten the lists
predictedcvRepeat = [x for xs in predictedcvRepeat for x in xs]
holdtrue = [x for xs in holdtrue for x in xs]

r2cv = metrics.r2_score(holdtrue, predictedcvRepeat)
print(r2cv)
figf, axf = plt.subplots()
lims = [
    np.min([holdtrue, predictedcvRepeat]),  # min of both axes
    np.max([holdtrue, predictedcvRepeat]),  # max of both axes
]
axf.plot(lims, lims, marker="o", alpha=0.5)
axf.hist2d(holdtrue, predictedcvRepeat, bins=50, cmap='YlGn')

axf.set_aspect('equal')  # can also be equal
axf.set_xlabel('Predicted Accretion Rate (mm/yr)')
axf.set_ylabel('Observed Accretion Rate (mm/yr)')
axf.set_title('Heatmap 100x Repeated 5-Fold CV for Whole Dataset')
axf.text(0.05, 0.95, str('R2: ' + str(round(r2cv, 4))), transform=axf.transAxes, fontsize=14,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
figf.show()
figf.savefig(RF2DtesttrainPlot + "wholeDataset.png")


def dataPredictRF(X_train, y_train, X_test, y_test):
    model = RandomForestRegressor(n_estimators=400, max_depth=90, random_state=0)
    model.fit(X_train, y_train)
    ypred = model.predict(X_test)
    bestMod = model
    # bestscore = 1
    bestscore = metrics.r2_score(ypred, y_test)
    return ypred, bestMod, bestscore  # there is no best score becuase there is no grid search


def dataPredictMLR(X_train, y_train, X_test, y_test):
    model = linear_model.LinearRegression()
    model.fit(X_train, y_train)
    ypred = model.predict(X_test)
    bestMod = model
    # bestscore = 1
    bestscore = metrics.r2_score(ypred, y_test)
    return ypred, bestMod, bestscore  # there is no best score becuase there is no grid search


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


# For random forest
bestscoresls = []
bestmodelsls = []
bestpredictedvalues = []
besttestvalues = []
besttrainvalues = []
besttestfortrain = []
for i in range(1, 251):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    ypred, bestmodel, bestscore = dataPredictRF(X_train, y_train, X_test, y_test)
    # Below for 2D

    bestscoresls.append(bestscore)
    bestmodelsls.append(bestmodel)
    bestpredictedvalues.append(ypred)
    besttestvalues.append(y_test)
    besttrainvalues.append(X_train)
    besttestfortrain.append(X_test)
    # if i == 1:
    #     y_obs = y_test
    #     y_predicted = ypred
    # else:
    #     y_obs = np.hstack((y_obs, y_test))
    #     y_predicted = np.hstack((y_predicted, ypred))

getidx = bestscoresls.index(np.max(bestscoresls))
print(bestscoresls[getidx])
# regression_results(y_predicted, y_obs)

# Plot best model results #
pred_acc = bestmodelsls[getidx].predict(besttestfortrain[getidx])
r2best = metrics.r2_score(bestpredictedvalues[getidx], besttestvalues[getidx])
print("R2 Best is ", r2best)
fig, ax = plt.subplots()
ax.scatter(bestpredictedvalues[getidx], besttestvalues[getidx])

lims = [
    np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
    np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
]

plt.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
ax.set_aspect('equal')  # can also be equal
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_title('Best RF Train-Test Prediction for Accretion for Whole Dataset ')
ax.set_xlabel('Predicted Accretion Rate (mm/yr)')
ax.set_ylabel('Observed Accretion Rate (mm/yr)')
ax.text(0.05, 0.95, str('R2: ' + str(round(r2best, 4))), transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
fig.show()
fig.savefig(RFtesttrainPlot + "wholeDataset.png")
print('################ BEST SPLIT REGRESSION RESULTS###############')
regression_results(bestpredictedvalues[getidx], besttestvalues[getidx])

# add SHAPLEY
bestmodel = bestmodelsls[getidx]
data = besttrainvalues[getidx]

shap_values = shap.TreeExplainer(bestmodel).shap_values(data)


def ABS_SHAP(df_shap, df, key):
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

    ax = k2.plot.barh(x='Variable', y='SHAP_abs', color=colorlist, figsize=(5, 6), legend=False)
    ax.set_xlabel("SHAP Value (Red = Positive Impact)")
    ax.set_title(str(key))
    figure = ax.get_figure()
    return figure


figure = ABS_SHAP(shap_values, data, "feat imp on whole dataset")
figure.show()
figure.savefig(featureimpPath + "wholeDataset.png")

# Make predictions for each of the marsh Class Datasets
import glob

path = "D:\\Etienne\\summer2022_CRMS\\everythingCRMS2\\experimentManyDatasets\\marshTypeDatasets"

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
    df = df.drop('Unnamed: 0.1', axis=1)
    df = df.rename(columns={'Unnamed: 0': 'Simple site'})
    listdfs.append(df)

cleandfs = {}
orderpairplots = []
allfeatsSplits = [
    'Distance_to_Bays_m', 'Distance_to_Ocean_m', 'Distance_to_Fluvial_m',
    'NDVI', 'tss_med', 'windspeed (m/s)', 'Land_Lost_m2',
    'Soil Salinity (ppt)', 'Average Height Herb (cm)',
    'avg_percentflooded (%)', 'Tide_Amp (ft)', 'avg_flooding (ft)', 'Flood Freq (Floods/yr)',
    outcome
]
for df in listdfs:
    dfname = df['marshType'][0]  # name of marsh type used to name dataframe and associated features
    cleandfs[dfname] = df[allfeatsSplits].dropna()  # extract the most important features identified for that dataframe

    # dictdf[dfname][outcome] = [np.exp(i) if i != 0 else 0 for i in dictdf[dfname][outcome]]  # because the mineral mass acc was logged prior to loading in

    cleandfs[dfname] = cleandfs[dfname].apply(pd.to_numeric)
    for col in cleandfs[dfname].columns.values:
        cleandfs[dfname][col + "_z"] = stats.zscore(
            cleandfs[dfname][col])  # Compute the zscore for each value per column ; save as "col"_z
    length = len(allfeats)
    for col in cleandfs[dfname].columns.values[
               length:]:  # 13 to 14 because I added the K_mean_marshType col for visualuization

        cleandfs[dfname] = cleandfs[dfname][np.abs(cleandfs[dfname][col]) < 3]  # keep if value is less than 2 std

    cleandfs[dfname] = cleandfs[dfname].drop(  # drop zscore columns
        cleandfs[dfname].columns.values[length:], axis=1)

meanscores = {}
maxscores = {}
semscores = {}
scoresdict = {}
for key in cleandfs:
    # evaluate a model with a given number of repeats (100x)
    print(key)
    d = cleandfs[key]  # extract df with only the most important features for the marsh type
    # Standardizing
    y = d[outcome]
    X = d.drop(outcome, axis=1)
    scaler = MinMaxScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns.values)

    results = list()
    # evaluate using a given number of repeats
    scores = evaluate_modelrf(X, y, 5, 100)
    # summarize
    print('>%d avg_score=%.4f se=%.3f' % (100, averagingfunc(scores), sem(scores)))
    scoresdict[key] = scores
    meanscores[key] = averagingfunc(scores)
    maxscores[key] = np.max(scores)
    semscores[key] = sem(scores)
    results.append(scores)

    #######

    predictedcvRepeat = []
    holdtrue = []
    holdmodels = []
    for r in range(1, 251):
        splits = KFold(n_splits=5, shuffle=True).split(X, y)
        model = get_rf().fit(X, y)
        predicted = cross_val_predict(model, X, y,
                                      cv=splits)  # RepeatedKFold(n_splits=5, n_repeats=100, random_state=1)
        predictedcvRepeat.append(predicted)
        holdtrue.append(y)
        holdmodels.append(model)
    # flatten the lists
    predictedcvRepeat = [x for xs in predictedcvRepeat for x in xs]
    holdtrue = [x for xs in holdtrue for x in xs]

    r2cv = metrics.r2_score(holdtrue, predictedcvRepeat)
    print(r2cv)
    figf, axf = plt.subplots()
    lims = [
        np.min([holdtrue, predictedcvRepeat]),  # min of both axes
        np.max([holdtrue, predictedcvRepeat]),  # max of both axes
    ]
    axf.plot(lims, lims, marker="o", alpha=0.5)
    axf.hist2d(holdtrue, predictedcvRepeat, bins=50, cmap='YlGn')

    axf.set_aspect('equal')  # can also be equal
    axf.set_xlabel('Predicted Accretion Rate (mm/yr)')
    axf.set_ylabel('Observed Accretion Rate (mm/yr)')
    axf.set_title('Heatmap 100x Repeated 5-Fold CV for ' + str(key))
    axf.text(0.05, 0.95, str('R2: ' + str(round(r2cv, 4))), transform=axf.transAxes, fontsize=14,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    figf.show()
    figf.savefig(RF2DtesttrainPlot + str(key) + ".png")

# Manual Runs
bestmodeldict = {}
bestscoredict = {}
bestpredictedDict = {}
besttestDict = {}
for key in cleandfs:
    print(key)
    d = cleandfs[key]  # extract df with only the most important features for the marsh type
    # Standardizing
    y = d[outcome]
    X = d.drop(outcome, axis=1)
    scaler = MinMaxScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns.values)

    bestscoresls = []
    bestmodelsls = []
    bestpredictedvalues = []
    besttestvalues = []
    besttrainvalues = []
    besttestfortrain = []
    for i in range(1, 251):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
        ypred, bestmodel, bestscore = dataPredictRF(X_train, y_train, X_test, y_test)
        bestscoresls.append(bestscore)
        bestmodelsls.append(bestmodel)
        bestpredictedvalues.append(ypred)
        besttestvalues.append(y_test)
        besttrainvalues.append(X_train)
        besttestfortrain.append(X_test)
        if i == 1:
            y_obs = y_test
            y_predicted = ypred
        else:
            y_obs = np.hstack((y_obs, y_test))
            y_predicted = np.hstack((y_predicted, ypred))

    # Plot best model results #
    getidx = bestscoresls.index(np.max(bestscoresls))
    pred_acc = bestmodelsls[getidx].predict(besttestfortrain[getidx])
    r2best = metrics.r2_score(bestpredictedvalues[getidx], besttestvalues[getidx])
    print("R2 Best is ", r2best)
    fig, ax = plt.subplots()
    ax.scatter(bestpredictedvalues[getidx], besttestvalues[getidx])

    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]

    plt.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    ax.set_aspect('equal')  # can also be equal
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_title('Best Random Train-Test Prediction for Accretion for ' + str(key))
    ax.set_xlabel('Predicted Accretion Rate (mm/yr)')
    ax.set_ylabel('Observed Accretion Rate (mm/yr)')
    ax.text(0.05, 0.95, str('R2: ' + str(round(r2best, 4))), transform=axf.transAxes, fontsize=14,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.show()
    fig.savefig(RFtesttrainPlot + str(key) + ".png")
    print('################ BEST SPLIT REGRESSION RESULTS###############')
    regression_results(bestpredictedvalues[getidx], besttestvalues[getidx])

    # add SHAPLEY for Feature Importance
    bestmodel = bestmodelsls[getidx]
    data = besttrainvalues[getidx]
    shap_values = shap.TreeExplainer(bestmodel).shap_values(data)
    # plot
    figure = ABS_SHAP(shap_values, data, str(key))
    figure.show()
    figure.savefig(featureimpPath + str(key) + ".png")

# Working on the concatenated datasets

path = "D:\\Etienne\\summer2022_CRMS\\everythingCRMS2\\experimentManyDatasets\\marshTypeDatasets\\accControlDatasets"

# csv files in the path
files = glob.glob(path + "/*.csv")
print(files)

# checking all the csv files in the
# specified path
listdfscontrols = []
for filename in files:
    # reading content of csv file
    # content.append(filename)
    df = pd.read_csv(filename, encoding="unicode_escape")
    # df = df.drop('Unnamed: 0', axis=1)
    df = df.rename(columns={'Unnamed: 0': 'Simple site'})
    listdfscontrols.append(df)

cleandfscontrols = {}

for df in listdfscontrols:
    dfname = df['marshControl'][0]  # name of marsh type used to name dataframe and associated features
    cleandfscontrols[dfname] = df[
        allfeatsSplits].dropna()  # extract the most important features identified for that dataframe

    # dictdf[dfname][outcome] = [np.exp(i) if i != 0 else 0 for i in dictdf[dfname][outcome]]  # because the mineral mass acc was logged prior to loading in

    cleandfscontrols[dfname] = cleandfscontrols[dfname].apply(pd.to_numeric)
    for col in cleandfscontrols[dfname].columns.values:
        cleandfscontrols[dfname][col + "_z"] = stats.zscore(
            cleandfscontrols[dfname][col])  # Compute the zscore for each value per column ; save as "col"_z
    length = len(allfeats)
    for col in cleandfscontrols[dfname].columns.values[
               length:]:  # 13 to 14 because I added the K_mean_marshType col for visualuization

        cleandfscontrols[dfname] = cleandfscontrols[dfname][
            np.abs(cleandfscontrols[dfname][col]) < 3]  # keep if value is less than 2 std

    cleandfscontrols[dfname] = cleandfscontrols[dfname].drop(  # drop zscore columns
        cleandfscontrols[dfname].columns.values[length:], axis=1)

meanscoresC = {}
maxscoreC = {}
semscoresC = {}
scoresdictC = {}
for key in cleandfscontrols:
    # evaluate a model with a given number of repeats (100x)
    print(key)
    d = cleandfscontrols[key]  # extract df with only the most important features for the marsh type
    # Standardizing
    y = d[outcome]
    X = d.drop(outcome, axis=1)
    scaler = MinMaxScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns.values)

    results = list()
    # evaluate using a given number of repeats
    scores = evaluate_modelrf(X, y, 5, 100)
    # summarize
    print('>%d avg_score=%.4f se=%.3f' % (100, averagingfunc(scores), sem(scores)))
    scoresdictC[key] = scores
    meanscoresC[key] = averagingfunc(scores)
    maxscoreC[key] = np.max(scores)
    semscoresC[key] = sem(scores)
    results.append(scores)

    ###################

    predictedcvRepeat = []
    holdtrue = []
    holdmodels = []
    for r in range(1, 251):
        splits = KFold(n_splits=5, shuffle=True).split(X, y)
        model = get_rf().fit(X, y)
        predicted = cross_val_predict(model, X, y,
                                      cv=splits)  # RepeatedKFold(n_splits=5, n_repeats=100, random_state=1)
        predictedcvRepeat.append(predicted)
        holdtrue.append(y)
        holdmodels.append(model)
    # flatten the lists
    predictedcvRepeat = [x for xs in predictedcvRepeat for x in xs]
    holdtrue = [x for xs in holdtrue for x in xs]

    r2cv = metrics.r2_score(holdtrue, predictedcvRepeat)
    print(r2cv)
    figf, axf = plt.subplots()
    lims = [
        np.min([holdtrue, predictedcvRepeat]),  # min of both axes
        np.max([holdtrue, predictedcvRepeat]),  # max of both axes
    ]
    axf.plot(lims, lims, marker="o", alpha=0.5)
    axf.hist2d(holdtrue, predictedcvRepeat, bins=50, cmap='YlGn')

    axf.set_aspect('equal')  # can also be equal
    axf.set_xlabel('Predicted Accretion Rate (mm/yr)')
    axf.set_ylabel('Observed Accretion Rate (mm/yr)')
    axf.set_title('Heatmap 100x Repeated 5-Fold CV for ' + str(key))
    axf.text(0.05, 0.95, str('R2: ' + str(round(r2cv, 4))), transform=axf.transAxes, fontsize=14,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    figf.show()
    figf.savefig(RF2DtesttrainPlot + str(key) + ".png")

# Manual Runs
bestmodeldictC = {}
bestscoredictC = {}
bestpredictedDictC = {}
besttestDictC = {}
for key in cleandfscontrols:
    print(key)
    d = cleandfscontrols[key]  # extract df with only the most important features for the marsh type
    # Standardizing
    y = d[outcome]
    X = d.drop(outcome, axis=1)
    scaler = MinMaxScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns.values)

    bestscoresls = []
    bestmodelsls = []
    bestpredictedvalues = []
    besttestvalues = []
    besttestfortrain = []
    for i in range(1, 251):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
        ypred, bestmodel, bestscore = dataPredictRF(X_train, y_train, X_test, y_test)
        bestscoresls.append(bestscore)
        bestmodelsls.append(bestmodel)
        bestpredictedvalues.append(ypred)
        besttestvalues.append(y_test)
        besttestfortrain.append(X_test)
        if i == 1:
            y_obs = y_test
            y_predicted = ypred
        else:
            y_obs = np.hstack((y_obs, y_test))
            y_predicted = np.hstack((y_predicted, ypred))

    # Plot best model results #
    getidx = bestscoresls.index(np.max(bestscoresls))
    pred_acc = bestmodelsls[getidx].predict(besttestfortrain[getidx])
    r2best = metrics.r2_score(bestpredictedvalues[getidx], besttestvalues[getidx])
    print("R2 Best is ", r2best)
    fig, ax = plt.subplots()
    ax.scatter(bestpredictedvalues[getidx], besttestvalues[getidx])

    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]

    plt.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    ax.set_aspect('equal')  # can also be equal
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_title('Best RF Train-Test Prediction for Accretion for ' + str(key))
    ax.set_xlabel('Predicted Accretion Rate (mm/yr)')
    ax.set_ylabel('Observed Accretion Rate (mm/yr)')
    ax.text(0.05, 0.95, str('R2: ' + str(r2best)),
            transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    fig.show()
    fig.savefig(RFtesttrainPlot + str(key) + ".png")
    print('################ BEST SPLIT REGRESSION RESULTS###############')
    regression_results(bestpredictedvalues[getidx], besttestvalues[getidx])

    # add SHAPLEY
    bestmodel = bestmodelsls[getidx]
    data = besttrainvalues[getidx]

    shap_values = shap.TreeExplainer(bestmodel).shap_values(data)
    # plot
    figure = ABS_SHAP(shap_values, data, str(key))
    figure.show()
    figure.savefig(featureimpPath + str(key) + ".png")
