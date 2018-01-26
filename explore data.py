import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

Xtrain = pd.read_csv('train_values.csv')
ytrain = pd.read_csv('train_labels.csv')
df = Xtrain.merge(ytrain, how = 'left', on = 'row_id')

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

if False:
    df['metro'] = df['area__rucc'].apply(lambda x: 'Nonmetro' if 'Nonmetro' in x else 'Metro')
    
    print(df.groupby(['metro'])['poverty_rate'].median())
    print(df.groupby(['metro'])['poverty_rate'].var())

if False:
    print(df[['health__pct_adult_obesity', 'health__pct_adult_smoking',
        'health__pct_diabetes', 'health__pct_excessive_drinking', 'poverty_rate']].corr())

if True:
    tempvar1 = df['demo__pct_aged_65_years_and_older'].median()
    df['metro'] = df['area__rucc'].apply(lambda x: 'Nonmetro' if 'Nonmetro' in x else 'Metro')
    df['old'] = df['demo__pct_aged_65_years_and_older'].apply(lambda x: 'elderly' if x > tempvar1 else 'not')
    print(df.groupby(['metro', 'old'])['poverty_rate'].median())
