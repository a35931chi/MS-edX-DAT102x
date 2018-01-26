import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from time import gmtime, strftime, time

from scipy import stats
from scipy.special import boxcox1p
from scipy.stats import norm, skew

from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split, GridSearchCV, ShuffleSplit
from sklearn.metrics import mean_squared_error

from sklearn.linear_model import ElasticNet, Lasso
import xgboost as xgb


Xtrain = pd.read_csv('train_values.csv')
Xtest = pd.read_csv('test_values.csv')
X = pd.concat([Xtrain, Xtest], axis = 0)
ytrain = pd.read_csv('train_labels.csv')
#boxcox performed the best
lam = 0.15
ytrain['poverty_rate'] = boxcox1p(ytrain['poverty_rate'] , lam)
df = X.merge(ytrain, how = 'left', on = 'row_id')
df.drop('row_id', axis = 1, inplace = True)

#some stat and data exploration
if False:
    print(ytrain.head())
    min(ytrain.poverty_rate)
    max(ytrain.poverty_rate)
    mean(ytrain.poverty_rate)
    np.median(ytrain.poverty_rate)
    np.std(ytrain.poverty_rate)
    
if False:
    plt.hist(ytrain.poverty_rate, bins = 10)
    plt.show()
    
if False:
    print(Xtrain.shape, ytrain.shape, df.shape)
    print(df.groupby(['econ__economic_typology'])['poverty_rate'].median())

    print(df.groupby(['area__urban_influence'])['poverty_rate'].median())
    print(df.groupby(['area__rucc'])['poverty_rate'].median())
    print(df.groupby(['yr'])['poverty_rate'].median())
    for col in ['area__rucc', 'area__urban_influence', 'econ__economic_typology', 'yr']:
        sns.boxplot(data = df, x = col, y = 'poverty_rate')
        plt.show()

if False:
    df['metro'] = df['area__rucc'].apply(lambda x: 'Nonmetro' if 'Nonmetro' in x else 'Metro')
    
    print(df.groupby(['metro'])['poverty_rate'].median())
    print(df.groupby(['metro'])['poverty_rate'].var())

if False:
    print(df[['health__pct_adult_obesity', 'health__pct_adult_smoking',
        'health__pct_diabetes', 'health__pct_excessive_drinking', 'poverty_rate']].corr())

if False:
    tempvar1 = df['demo__pct_aged_65_years_and_older'].median()
    df['metro'] = df['area__rucc'].apply(lambda x: 'Nonmetro' if 'Nonmetro' in x else 'Metro')
    df['old'] = df['demo__pct_aged_65_years_and_older'].apply(lambda x: 'elderly' if x > tempvar1 else 'not')
    print(df.groupby(['metro', 'old'])['poverty_rate'].median())

#for speed purposes, I'm going to use a quick regression method
#feature engineering

if False: 
        lam = 0.15
        fig, (ax1, ax2, ax3) = plt.subplots(ncols = 3, figsize = (15,6))
        sns.distplot(df['poverty_rate'], fit = norm, ax = ax1)
        sns.distplot(boxcox1p(df['poverty_rate'], lam), fit = norm, ax = ax2)
        sns.distplot(np.log1p(df['poverty_rate']), fit = norm, ax = ax3)
        # Get the fitted parameters used by the function
        (mu1, sigma1) = norm.fit(df['poverty_rate'])
        (mu2, sigma2) = norm.fit(boxcox1p(df['poverty_rate'], lam))
        (mu3, sigma3) = norm.fit(np.log1p(df['poverty_rate']))
        ax1.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu1, sigma1),
                    'Skewness: {:.2f}'.format(skew(df['poverty_rate']))],
                    loc = 'best')
        ax2.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu2, sigma2),
                    'Skewness: {:.2f}'.format(skew(boxcox1p(df['poverty_rate'], lam)))],
                    loc = 'best')
        ax3.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu2, sigma3),
                    'Skewness: {:.2f}'.format(skew(np.log1p(df['poverty_rate'])))],
                    loc = 'best')
        ax1.set_ylabel('Frequency')
        ax1.set_title('poverty_rate Distribution')
        ax2.set_title('poverty_rate Box-Cox Transformed')
        ax3.set_title('poverty_rate Log Transformed')
        plt.tight_layout()
        plt.show()

        #Get also the QQ-plot
        if True:
            fig = plt.subplots(figsize = (15,6))
            ax1 = plt.subplot(131)
            res = stats.probplot(df['poverty_rate'], plot = plt)
            ax1.set_title('poverty_rate Probability Plot')
            
            ax2 = plt.subplot(132)
            res = stats.probplot(boxcox1p(df['poverty_rate'], lam), plot = plt)
            ax2.set_title('poverty_rate Box-Cox Transformed Probability Plot')
            
            ax3 = plt.subplot(133)
            res = stats.probplot(np.log1p(df['poverty_rate']), plot = plt)
            ax3.set_title('poverty_rate Log Transformed Probability Plot')
            
            plt.tight_layout()
            plt.show()



#get some stats from the data as a whole
if False:
    all_data_na = (df.isnull().sum() / len(df)) * 100
    all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)
    missing_data = pd.DataFrame({'Missing Ratio': all_data_na})
        
    f, ax = plt.subplots(figsize=(10, 7))
    plt.xticks(rotation = '90')
    sns.barplot(x = all_data_na.index, y = all_data_na)
    plt.xlabel('Features', fontsize = 15)
    plt.ylabel('Percent of missing values', fontsize = 15)
    plt.title('Percent NaNs by Features', fontsize = 15)
    plt.tight_layout()
    plt.savefig('Percent NaNs by Features.png')
    plt.show()

    '''
                                             Missing Ratio
health__homicides_per_100k                       61.507192  negative doesn't make sense, empties should be median
health__pct_excessive_drinking                   30.581614  empties should be median
health__pct_adult_smoking                        14.509068  empties should be median
health__motor_vehicle_crash_deaths_per_100k      13.039400  empties should be median
health__pop_per_dentist                           7.629769  empties should be median
health__pop_per_primary_care_physician            7.191995  empties should be median
health__pct_low_birthweight                       5.691057  empties should be median
health__air_pollution_particulate_matter          0.875547  empties should be median
demo__pct_non_hispanic_african_american           0.062539  empties should be median
econ__pct_uninsured_children                      0.062539  empties should be median
demo__pct_female                                  0.062539  empties should be median
demo__pct_below_18_years_of_age                   0.062539  empties should be median
demo__pct_aged_65_years_and_older                 0.062539  empties should be median
demo__pct_hispanic                                0.062539  empties should be median
health__pct_adult_obesity                         0.062539  empties should be median
demo__pct_non_hispanic_white                      0.062539  empties should be median
demo__pct_american_indian_or_alaskan_native       0.062539  empties should be median
demo__pct_asian                                   0.062539  empties should be median
health__pct_diabetes                              0.062539  empties should be median
health__pct_physical_inacticity                   0.062539  empties should be median
econ__pct_uninsured_adults                        0.062539  empties should be median
    '''

cat_cols = ['area__rucc', 'area__urban_influence', 'econ__economic_typology', 'yr']
num_cols = [col for col in df.columns.values if col not in cat_cols and col != 'poverty_rate']

for col in num_cols:
    df[col] = df.groupby(['area__rucc', 'area__urban_influence', 'econ__economic_typology', 'yr'])[col].transform(lambda x: x.fillna(x.median()))
''' still missing
                                             Missing Ratio
health__homicides_per_100k                       24.358974
health__pct_excessive_drinking                    1.188243
health__pop_per_dentist                           0.250156
health__motor_vehicle_crash_deaths_per_100k       0.250156
health__pct_adult_smoking                         0.187617
health__air_pollution_particulate_matter          0.125078
health__pop_per_primary_care_physician            0.062539
health__pct_low_birthweight                       0.031270
'''
for col in num_cols:
    df[col] = df.groupby(['area__rucc', 'area__urban_influence', 'econ__economic_typology'])[col].transform(lambda x: x.fillna(x.median()))
''' still missing
                                             Missing Ratio
health__homicides_per_100k                       22.138837
health__pct_excessive_drinking                    1.188243
health__pop_per_dentist                           0.250156
health__pct_adult_smoking                         0.187617
health__air_pollution_particulate_matter          0.125078
health__pop_per_primary_care_physician            0.062539
health__motor_vehicle_crash_deaths_per_100k       0.062539
'''
for col in num_cols:
    df[col] = df.groupby(['area__rucc', 'area__urban_influence'])[col].transform(lambda x: x.fillna(x.median()))

''' still missing
                                Missing Ratio
health__homicides_per_100k           8.567855
health__pct_excessive_drinking       0.250156
'''

for col in num_cols:
    df[col] = df.groupby(['area__rucc'])[col].transform(lambda x: x.fillna(x.median()))



if False:
    all_data_na = (df.isnull().sum() / len(df)) * 100
    all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)
    missing_data = pd.DataFrame({'Missing Ratio': all_data_na})
        
    f, ax = plt.subplots(figsize=(10, 7))
    plt.xticks(rotation = '90')
    sns.barplot(x = all_data_na.index, y = all_data_na)
    plt.xlabel('Features', fontsize = 15)
    plt.ylabel('Percent of missing values', fontsize = 15)
    plt.title('Percent NaNs by Features', fontsize = 15)
    plt.tight_layout()
    plt.savefig('Percent NaNs by Features.png')
    plt.show()
    print(missing_data)

if False: 
    for col in num_cols:    
        fig, (ax1, ax2, ax3) = plt.subplots(ncols = 3, figsize = (15,6))
        sns.distplot(df[col], fit = norm, ax = ax1)
        sns.distplot(boxcox1p(df[col], lam), fit = norm, ax = ax2)
        sns.distplot(np.log1p(df[col]), fit = norm, ax = ax3)
        # Get the fitted parameters used by the function
        (mu1, sigma1) = norm.fit(df[col])
        (mu2, sigma2) = norm.fit(boxcox1p(df[col], lam))
        (mu3, sigma3) = norm.fit(np.log1p(df[col]))
        ax1.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu1, sigma1),
                    'Skewness: {:.2f}'.format(skew(df[col]))],
                    loc = 'best')
        ax2.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu2, sigma2),
                    'Skewness: {:.2f}'.format(skew(boxcox1p(df[col], lam)))],
                    loc = 'best')
        ax3.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu2, sigma3),
                    'Skewness: {:.2f}'.format(skew(np.log1p(df[col])))],
                    loc = 'best')
        ax1.set_ylabel('Frequency')
        ax1.set_title(col + ' Distribution')
        ax2.set_title(col + ' Box-Cox Transformed')
        ax3.set_title(col + ' Log Transformed')
        plt.tight_layout()
        plt.show()

        #Get also the QQ-plot
        if False:
            fig = plt.subplots(figsize = (15,6))
            ax1 = plt.subplot(131)
            res = stats.probplot(df[col], plot = plt)
            ax1.set_title(col + ' Probability Plot')
            
            ax2 = plt.subplot(132)
            res = stats.probplot(boxcox1p(df[col], lam), plot = plt)
            ax2.set_title(col + ' Box-Cox Transformed Probability Plot')
            
            ax3 = plt.subplot(133)
            res = stats.probplot(np.log1p(df[col]), plot = plt)
            ax3.set_title(col + ' Log Transformed Probability Plot')
            
            plt.tight_layout()
            plt.show()
'''
log transform:
econ__pct_civilian_labor
econ__pct_unemployment
econ__pct_uninsured_adults
econ__pct_uninsured_children
demo__pct_below_18_years_of_age
demo__pct_aged_65_years_and_older
demo__pct_hispanic
demo__pct_non_hispanic_african_american
demo__pct_american_indian_or_alaskan_native
demo__pct_asian
demo__pct_adults_less_than_a_high_school_diploma
demo__pct_adults_bachelors_or_higher
health__pct_adult_smoking
health__pct_diabetes
health__pct_low_birthweight
health__pct_excessive_drinking
health__pop_per_dentist
health__pop_per_primary_care_physician

boxcox:
demo__birth_rate_per_1k
health__homicides_per_100k
health__motor_vehicle_crash_deaths_per_100k

no transform:
demo__pct_female
demo__pct_non_hispanic_white
demo__pct_adults_with_high_school_diploma
demo__pct_adults_with_some_college
demo__death_rate_per_1k
health__pct_adult_obesity
health__pct_physical_inacticity
health__air_pollution_particulate_matter
'''

log_trans_cols = ['econ__pct_civilian_labor', 'econ__pct_unemployment', 'econ__pct_uninsured_adults',
                  'econ__pct_uninsured_children', 'demo__pct_below_18_years_of_age',
                  'demo__pct_aged_65_years_and_older', 'demo__pct_hispanic',
                  'demo__pct_non_hispanic_african_american', 'demo__pct_american_indian_or_alaskan_native',
                  'demo__pct_asian', 'demo__pct_adults_less_than_a_high_school_diploma',
                  'demo__pct_adults_bachelors_or_higher', 'health__pct_adult_smoking',
                  'health__pct_diabetes', 'health__pct_low_birthweight',
                  'health__pct_excessive_drinking', 'health__pop_per_dentist',
                  'health__pop_per_primary_care_physician']

boxcox_trans_cols = ['demo__birth_rate_per_1k', 'health__homicides_per_100k',
                     'health__motor_vehicle_crash_deaths_per_100k']

for col in log_trans_cols:
    df[col] = np.log1p(df[col])

for col in boxcox_trans_cols:
    df[col] = boxcox1p(df[col], lam)
    
if False: #correlation X vs. boxcox
        corrmat = df.corr()
        plt.subplots(figsize = (12, 9))
        g = sns.heatmap(corrmat, vmax = 0.9, square = True)
        g.set_yticklabels(g.get_yticklabels(), rotation = 0, fontsize = 8)
        g.set_xticklabels(g.get_xticklabels(), rotation = 90, fontsize = 8)
        plt.title('Correlation Matrix/Heatmap Numerical Features vs. boxcox')
        plt.tight_layout()
        plt.savefig('Numerical Features vs. boxcox heatmap.png')
        plt.show()

    # process columns, apply LabelEncoder to categorical features
for col in cat_cols:
    lbl = LabelEncoder() 
    lbl.fit(list(df[col].values)) 
    df[col] = lbl.transform(list(df[col].values))

#scaler = RobustScaler()
#Xscaled = scaler.fit_transform(df.drop('poverty_rate', axis = 1))
Xtrain = df[~np.isnan(df['poverty_rate'])].drop('poverty_rate', axis = 1)
Xtest = df[np.isnan(df['poverty_rate'])].drop('poverty_rate', axis = 1)



Xtrain, Xval, ytrain, yval = train_test_split(Xtrain, ytrain.drop('row_id', axis = 1), test_size=0.3, random_state=42)

scaler = RobustScaler()
Xtrain_scaled = scaler.fit_transform(Xtrain)
Xval_scaled = scaler.transform(Xval)
Xtest_scaled = scaler.transform(Xtest)


print(Xtrain_scaled.shape, ytrain.shape, Xval_scaled.shape, yval.shape, Xtest_scaled.shape)
what = input('wait')

def rmse(prediction, yval): #this method calculates the metrics
    return np.sqrt(mean_squared_error(prediction, yval))

def Lasso_GSCV(Xtrain, Xval, ytrain, yval):
    '''

    '''
    print('Lasso GridSearchCV: ', strftime('%d %b %Y %H:%M:%S', gmtime()))
    params = {'alpha': [0.0001, 0.00012, 0.00014, 0.00016, 0.00018, 0.0002],
              'max_iter': range(100, 2000, 100)}
    
    cv_sets = ShuffleSplit(n_splits = 5, test_size = 0.2)
    t0 = time()

    scaler = RobustScaler()
    Xtrainscaled = scaler.fit_transform(Xtrain)
    Xvalscaled = scaler.transform(Xval)

    regressor = Lasso()
    grid = GridSearchCV(estimator = regressor, param_grid = params,
                        scoring = 'neg_mean_squared_error', cv = cv_sets)
    
    grid = grid.fit(Xtrainscaled, ytrain)
    
    test_score = rmse(grid.predict(Xvalscaled), yval)
    print('Time algo takes: {:.3f} seconds'.format(time() - t0))
    print('Train error: {:.4f} ({:.2f}%)'.format(np.sqrt(-grid.best_score_), np.sqrt(-grid.best_score_) / np.mean(ytrain) * 100))
    print('Test error: {:.4f} ({:.2f}%)'.format(test_score, test_score / np.mean(yval) * 100))
    
    print(grid.best_estimator_)
    #print(grid.cv_results_)
    pass


def ENetR_GSCV(Xtrain, Xval, ytrain, yval):
    '''            
    Time algo takes: 73.447 seconds
    Train error: 0.2749 (7.81%)
    Test error: 0.2693 (7.69%)
    ElasticNet(alpha=8e-05, copy_X=True, fit_intercept=True, l1_ratio=8,
          max_iter=50, normalize=False, positive=False, precompute=False,
          random_state=None, selection='cyclic', tol=0.0001, warm_start=False)
      
    '''
    print('ENet GridSearchCV: ', strftime('%d %b %Y %H:%M:%S', gmtime()))
    params = {'alpha': [0.00008, 0.00009, 0.0001, 0.00011, 0.00012],
              'l1_ratio': range(1, 20, 1),
              'max_iter': range(50, 850, 50)}
    
    cv_sets = ShuffleSplit(n_splits = 5, test_size = 0.2)
    
    t0 = time()
    
    #scaler = RobustScaler()
    #Xtrainscaled = scaler.fit_transform(Xtrain)
    #Xvalscaled = scaler.transform(Xval)
    
    regressor = ElasticNet()
    what = input('wait1')
    grid = GridSearchCV(estimator = regressor, param_grid = params,
                        scoring = 'neg_mean_squared_error', cv = cv_sets)
    what = input('wait2')
    grid = grid.fit(Xtrain, ytrain)
    what = input('wait3')
    test_score = rmse(grid.predict(Xval), yval)
    what = input('wait4')
    print('Time algo takes: {:.3f} seconds'.format(time() - t0))
    what = input('wait5')
    print(-grid.best_score_)
    print('Train error: {:.4f} ({:.2f}%)'.format(np.sqrt(-grid.best_score_), np.sqrt(-grid.best_score_) / np.mean(ytrain) * 100))
    what = input('wait6')
    print('Test error: {:.4f} ({:.2f}%)'.format(test_score, test_score / np.mean(yval) * 100))
    what = input('wait7')
    
    print(grid.best_estimator_)
    #print(grid.cv_results_)
    
    #return grid, scaler
    pass

def XGBR_GSCV(Xtrain, Xval, ytrain, yval):
    '''

    '''
    print('XGBoost GridSearchCV: ', strftime('%d %b %Y %H:%M:%S', gmtime()))
    params = {'learning_rate': [0.0001, 0.001, 0.01, 0.1, 1],
              'max_depth': range(1, 11, 1),
              'n_estimators': range(1, 20, 2), 
              'reg_alpha': [0.0001, 0.001, 0.01, 0.1, 1]}
    
    cv_sets = ShuffleSplit(n_splits = 5, test_size = 0.2)
    regressor = xgb.XGBRegressor()
    t0 = time()
    grid = GridSearchCV(estimator = regressor, param_grid = params,
                        scoring = 'neg_mean_squared_error', cv = cv_sets)
    
    grid = grid.fit(Xtrain, ytrain)

    test_score = rmse(grid.predict(Xval), yval)
    print('Time algo takes: {:.3f} seconds'.format(time() - t0))
    print('Train error: {:.4f} ({:.2f}%)'.format(np.sqrt(-grid.best_score_), np.sqrt(-grid.best_score_) / np.mean(ytrain) * 100))
    print('Test error: {:.4f} ({:.2f}%)'.format(test_score, test_score / np.mean(yval) * 100))
    
    print(grid.best_estimator_)
    #print(grid.cv_results_)
    pass

#Lasso_GSCV(Xtrain, Xval, ytrain, yval)
#model, scaler = ENetR_GSCV(Xtrain, Xval, ytrain, yval)
#XGBR_GSCV(Xtrain, Xval, ytrain, yval)

ENetR_GSCV(Xtrain_scaled, Xval_scaled, ytrain, yval)

#prediction = grid.predict(Xtest)
print('done')
