

# https://github.com/timestocome

# data source
# http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic3.xls
# http://campus.lakeforest.edu/frank/FILES/MLFfiles/Bio150/Titanic/TitanicMETA.pdf



import pandas as pd
import numpy as np
import re




# read in csv file of data
data = pd.read_csv('titanic3.csv')
columns = data.columns.values



# print out some basic infomation about data
pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 60)

print(data.head())
print(data.columns.values)
print(data.dtypes)



# row 1309 is filled with NaN
data = data.drop(data.index[[1309]])



# convert pclass to one hot vector
data['Class 1'] = np.where(data['pclass'] == 1., 1, 0)
data['Class 2'] = np.where(data['pclass'] == 2., 1, 0)
data['Class 3'] = np.where(data['pclass'] == 3., 1, 0)
data = data.drop('pclass', axis=1)


# convert sex to one hot vector
data['Sex'] = np.where(data['sex'] == 'male', 1, 0)
data = data.drop('sex', axis=1)


# convert survived to ints
data['Survived'] = np.where(data['survived'] == 1., 1, 0)
data = data.drop('survived', axis=1)
data = data.drop('body', axis=1)  # redundent 
data = data.drop('boat', axis=1)


# convert sibsp to T/F  Sibling and or Spouse
data['Relatives'] = np.where(data['sibsp'] > 0, 1, 0)
data = data.drop('sibsp', axis=1)

# convert parch to T/F Parent and or Child
data['Parent_Child'] = np.where(data['parch'] > 0, 1, 0)
data = data.drop('parch', axis=1)

# convert embarked location to one hots
data['S'] = np.where(data['embarked'] == 'S', 1, 0)
data['C'] = np.where(data['embarked'] == 'C', 1, 0)
data['Q'] = np.where(data['embarked'] == 'Q', 1, 0)
data = data.drop('embarked', axis=1)


# convert names into useful data
# pull titles from names, convert to one hot
def get_title(name):
    title = re.search(' ([A-Za-z]+)\.', name)
    
    if title:
        return title.group(1)
    else:
        return 'No title'

data['Title'] = data['name'].apply(get_title)


titles = data['Title'].value_counts()
# print(titles)

data['Mr'] = np.where(data['Title'] == 'Mr', 1, 0)
data['Miss'] = np.where(data['Title'] == 'Miss', 1, 0)
data['Mrs'] = np.where(data['Title'] == 'Mrs', 1, 0)
data['Master'] = np.where(data['Title'] == 'Master', 1, 0)
data['Rev'] = np.where(data['Title'] == 'Rev', 1, 0)
data['Dr'] = np.where(data['Title'] == 'Dr', 1, 0)
data['Col'] = np.where(data['Title'] == 'Col', 1, 0)
data['Major'] = np.where(data['Title'] == 'Major', 1, 0)
data['Mlle'] = np.where(data['Title'] == 'Mlle', 1, 0)
data['Ms'] = np.where(data['Title'] == 'Ms', 1, 0)
data['Capt'] = np.where(data['Title'] == 'Capt', 1, 0)
data['Dona'] = np.where(data['Title'] == 'Dona', 1, 0)
data['Don'] = np.where(data['Title'] == 'Don', 1, 0)
data['Countess'] = np.where(data['Title'] == 'Countess', 1, 0)
data['Sir'] = np.where(data['Title'] == 'Sir', 1, 0)
data['Jonkheer'] = np.where(data['Title'] == 'Jonkheer', 1, 0)
data['Mme'] = np.where(data['Title'] == 'Mme', 1, 0)
data['Lady'] = np.where(data['Title'] == 'Lady', 1, 0)


data = data.drop('name', axis=1)


# lots of NaN in age use median value for title
median_age = data.groupby('Title').median()  # group by titles, get median values
median_age = median_age['age']   # lookup table for age by title
# print(median_age)
 
# list of passengers with missing ages, replace Nan with Title
missing_ages = data.loc[data.age.isnull(), 'Title']
#print(missing_ages)

# replace titles with median age
data.loc[data.age.isnull(), 'age'] = median_age.loc[missing_ages].values
data = data.drop('Title', axis=1)

# scale age
data['age'] = data['age'] / data['age'].max()
#print('max age', data['age'].max())



# get deck letter from ticket, convert to one hot
def get_deck(cabin):
    if pd.isnull(cabin):
        return 'Z'
    return cabin[0]


data['Deck'] = data['cabin'].apply(get_deck)
decks = data['Deck'].value_counts()
#print(decks)

# lots of nulls only one hot the ones with values
data['Deck C'] = np.where(data['Deck'] == 'C', 1, 0)
data['Deck B'] = np.where(data['Deck'] == 'B', 1, 0)
data['Deck D'] = np.where(data['Deck'] == 'D', 1, 0)
data['Deck E'] = np.where(data['Deck'] == 'E', 1, 0)
data['Deck A'] = np.where(data['Deck'] == 'A', 1, 0)
data['Deck F'] = np.where(data['Deck'] == 'F', 1, 0)
data['Deck G'] = np.where(data['Deck'] == 'G', 1, 0)
data['Deck T'] = np.where(data['Deck'] == 'T', 1, 0)
data = data.drop('Deck', axis=1)


# bucket fares

# bin data to smooth outliers
# 3 Classes - try 6 bins min 0, max 512, meds: 8.05, 15.05, 60.0
# 0-8, 8-11.5, 11.5-15, 15-37.5, 37.5-60, 60-512
# Note: edges <> so go a bit high/low to cover ==
bins = [-1, 8, 11.5, 15, 37.5, 60, 513]
labels = ['Lower 3rd', 'Upper 3rd', 'Lower 2nd', 'Upper 2nd', 'Lower 1st', 'Upper 1st']
data['fare_categories'] = pd.cut(data['fare'], bins, labels=labels)

data['Lower 3rd'] = np.where(data['fare_categories'] == 'Lower 3rd', 1, 0)
data['Upper 3rd'] = np.where(data['fare_categories'] == 'Upper 3rd', 1, 0)
data['Lower 2nd'] = np.where(data['fare_categories'] == 'Lower 2nd', 1, 0)
data['Upper 2nd'] = np.where(data['fare_categories'] == 'Upper 2nd', 1, 0)
data['Lower 1st'] = np.where(data['fare_categories'] == 'Lower 1st', 1, 0)
data['Upper 1st'] = np.where(data['fare_categories'] == 'Upper 1st', 1, 0)
data = data.drop('fare_categories', axis=1)


# convert output to one hot
data['Drowned'] = np.where(data['Survived'] == 0, 1, 0)




data = data.drop('home.dest', axis=1)
data = data.drop('ticket', axis=1)
data = data.drop('fare', axis=1)
data = data.drop('cabin', axis=1)

data.to_csv('cleaned_titanic.csv')

print(data)

