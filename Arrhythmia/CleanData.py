#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 13:45:09 2018

# http://github.com/timestocome
# Linda MacPhee-Cobb


"""



################################################################################
# read in data files
# arrhythmia.data
# arrhythmia.names


'''
 The aim is to determine the type of arrhythmia from 
      the ECG recordings.
 This database contains 279 attributes, 206 of which are linear
     valued and the rest are nominal. 

     Concerning the study of H. Altay Guvenir: "The aim is to distinguish
     between the presence and absence of cardiac arrhythmia and to
     classify it in one of the 16 groups. Class 01 refers to 'normal'
     ECG classes 02 to 15 refers to different classes of arrhythmia
     and class 16 refers to the rest of unclassified ones. For the
     time being, there exists a computer program that makes such a
     classification. However there are differences between the
     cardiolog's and the programs classification. Taking the
     cardiolog's as a gold standard we aim to minimise this difference
     by means of machine learning tools."

     The names and id numbers of the patients were recently 
     removed from the database.
'''     
     

# 16 classes
# 01: 245
# 02: 44
# 03: 15
# 04: 15
# 05: 13
# 05: 25
# 06: 3
# 07: 2
# 08: 9
# 09: 50
# 10: 0
# 11: 0
# 12: 0
# 13: 0
# 14: 4
# 15: 5
# 16: 22
###############################################################################
import numpy as np
import pandas as pd
#pd.options.display.max_rows = 320
#pd.options.display.max_columns = 320

data = pd.read_csv('arrhythmia.data', header=None)

feature_names = ['Age', 'Sex', 'Height', 'Weight', 'QRS duration', 'P_R', 'Q_T', 'T_Interval', 
                 'P_Interval', 'QRS', 'T', 'P', 'QRST', 'J', 'BPM', 'Q width', 
                 'R width', 'S width', 'R_prime width', 'S_prime width', 'N deflections',
                 'Ragged R', 'Diphasic R', 'Ragged P', 'Diphasic P', 'Ragged T', 
                 'Diphasic T', 'DII 28', 'DII 29', 'DII 30', 'DII 31', 'DII 32', 
                 'DII 33', 'DII 34', 'DII 35', 'DII 36', 'DII 37', 'DII 38', 'DII 39', 
                 'DIII 40', 'DIII 41', 'DIII 42', 'DIII 43', 'DIII 44', 'DIII 45', 
                 'DIII 46', 'DIII 47', 'DIII 48', 'DIII 49', 'DIII 50', 'DIII 51', 
                 'AVR 52', 'AVR 53', 'AVR 54', 'AVR 55', 'AVR 56', 'AVR 57', 'AVR 58',
                 'AVR 59', 'AVR 60', 'AVR 61', 'AVR 62', 'AVR 63', 'AVL 64', 'AVL 65',
                 'AVL 66', 'AVL 67', 'AVL 68', 'AVL 69', 'AVL 70', 'AVL 71', 'AVL 72',
                 'AVL 73', 'AVL 74', 'AVL 75', 'AVF 76', 'AVF 77', 'AVF 78', 'AVF 79', 
                 'AVF 80', 'AVF 81', 'AVF 82', 'AVF 83', 'AVF 84', 'AVF 85', 'AVF 86',
                 'AVF 87', 'V1 88', 'V1 89', 'V1 90', 'V1 91', 'V1 92', 'V1 93', 'V1 94',
                 'V1 95', 'V1 96', 'V1 97', 'V1 98', 'V1 99', 'V2 100', 'V2 101', 
                 'V2 102', 'V2 103', 'V2 104', 'V2 105', 'V2 106', 'V2 107', 'V2 108',
                 'V2 109', 'V2 110', 'V2 111', 'V3 112', 'V3 113', 'V3 114', 'V3 115', 
                 'V3 116', 'V3 117', 'V3 118', 'V3 119', 'V3 120', 'V3 121', 'V3 122', 
                 'V3 123', 'V4 124', 'V4 125', 'V4 126', 'V4 127', 'V4 128', 'V4 129', 
                 'V4 130', 'V4 131', 'V4 132', 'V4 133', 'V4 134', 'V4 135', 'V5 136', 
                 'V5 137', 'V5 138', 'V5 139', 'V5 140', 'V5 141', 'V5 142', 'V5 143', 
                 'V5 144', 'V5 145', 'V5 146', 'V5 147', 'V6 148', 'V6 149', 'V6 150', 
                 'V6 151', 'V6 152', 'V6 153', 'V6 154', 'V6 155', 'V6 156', 'V6 157', 
                 'V6 158', 'V6 159', 'JJ_wave', 'Q_wave', 'R_wave', 'S_wave', 'R prime wave', 
                 'S prime wave', 'P wave', 
                 'T wave', 'QRSA', 'QRSTA', 'DII 170', 'DII 171', 'DII 172', 'DII 173', 
                 'DII 174', 'DII 175', 'DII 176', 'DII 177', 'DII 178', 'DII 179', 
                 'DIII 180', 'DIII 181', 'DIII 182', 'DIII 183', 'DIII 184', 'DIII 185', 
                 'DIII 186', 'DIII 187', 'DIII 188', 'DIII 189', 'AVR 190', 'AVR 191',
                 'AVR 192', 'AVR 193', 'AVR 194', 'AVR 195', 'AVR 196', 'AVR 197', 
                 'AVR 198', 'AVR 199', 'AVL 200', 'AVL 201', 'AVL 202', 'AVL 203', 
                 'AVL 204', 'AVL 205', 'AVL 206', 'AVL 207', 'AVL 208', 'AVL 209', 
                 'AVF 210', 'AVF 211', 'AVF 212', 'AVF 213', 'AVF 214', 'AVF 215', 
                 'AVF 216', 'AVF 217', 'AVF 218', 'AVF 219', 'V1 220', 'V1 221', 
                 'V1 222', 'V1 223', 'V1 224', 'V1 225', 'V1 226', 'V1 227', 'V1 228', 
                 'V1 229', 'V2 230', 'V2 231', 'V2 232', 'V2 233', 'V2 234', 'V2 235', 
                 'V2 236', 'V2 237', 'V2 238', 'V2 239', 'V3 240', 'V3 241', 'V3 242', 
                 'V3 243', 'V3 244', 'V3 245', 'V3 246', 'V3 247', 'V3 248', 'V3 249', 
                 'V4 250', 'V4 251', 'V4 252', 'V4 253', 'V4 254', 'V4 255', 'V4 256', 
                 'V4 257', 'V4 258', 'V4 259', 'V5 260', 'V5 261', 'V5 262', 'V5 263', 
                 'V5 264', 'V5 265', 'V5 266', 'V5 267', 'V5 268', 'V5 269', 'V6 270', 
                 'V6 271', 'V6 272', 'V6 273', 'V6 274', 'V6 275', 'V6 276', 'V6 277', 
                 'V6 278', 'V6 279', 'Class'
                 ]


# label columns to make things easier
data.columns = feature_names





###############################################################################
# there are no samples in dataset with these Classes - 
# drop them to reduce state space, it's a small dataset
# 16 classes of problems, only 13 in this dataset
###############################################################################


classes = [ 'Normal', 'Ischemic changes (CAD)',  'Old Anterior Myocardial Infarction',
           'Old Inferior Myocardial Infarction', 'Supraventricular Premature Contraction',
          'Left bundle branch block', 'Right bundle branch block',	
          '1. degree AtrioVentricular block',	 '2. degree AV block',	
          '3. degree AV block',	  'Left ventricule hypertrophy', 
          'Atrial Fibrillation or Flutter',  'Others']

print(len(classes))











# take a quick look at things

n_samples = 451

'''
print(data)
print('data cols', len(data.columns), len(feature_names))
print('data rows', len(data))
'''

# check for columns of zeros
# V6 275, V5 265, AVL 205, S prime, V6 157, V6 158, V6 152, V5 146, V5 144,
# V5 142, V5 140, V4 133, V4 132, AVF 84, AVL 70, AVL 68, S prime width, 
#print(data.sum())

# double check 
#print(data.describe())


# remove zero columns
data.drop( ['V6 275', 'V5 265', 'AVL 205', 'S prime wave', 'V6 157', 'V6 158', 
                  'V6 152', 'V5 146', 'V5 144', 'V5 142', 'V5 140', 'V4 133', 'V4 132', 
                  'AVF 84', 'AVL 70', 'AVL 68', 'S_prime width'], axis=1, inplace=True)




# check for nan and remove, if any. None in this dataset
#print(data.isna().sum())

# drop columns of objects with '?' instead of data - hopefully we won't need them
#print(data.dtypes)
data.drop(['T', 'P', 'QRST', 'J', 'BPM'], axis=1, inplace=True)




# see what's left

target = ['Class']

# see which features we still have
features = list(data.columns.values)
features.remove('Class')



###############################################################################
# there's not much data here and there are lots of classes and features
# let's save the orignal data as a holdout set to validate the madel and 
# generate some training data
###############################################################################
print('-----  original data ----------------')
print(data.head())
print(data.describe())




print('-----   create training data  ------------')

def create_noise(series, n):

    mu = series.mean()
    sigma = series.std()

    return np.random.normal(mu, sigma, n) * 0.01


# copy training data
train_data = pd.DataFrame(np.repeat(data.values, 10, axis=0))
train_data.columns = data.columns.values

# add noise to training data
cols = train_data.columns.values


n_train = len(train_data)

scale_features = features
scale_features.remove('Sex')
for f in features:
    train_data[f] += create_noise(train_data[f], n_train)


print('train_data should look like original data')
print('number of training samples', len(train_data))
print(train_data.describe())





###############################################################################
# scale, split them write to disk
###############################################################################

def split_data(db):
    x = db[features]
    y = db[target]
    
    return x, y


x, y = split_data(data)
x_train, y_train = split_data(train_data)





# you are here ***********************************************************
# don't scale categories ( sex ) or columns that already max out at 1


# check value range by feature, 
# scale any features that have values less than zero or greater than 1
for i in features:
   
    f_min = x[i].min()
    f_max = x[i].max()
    
    
    if (f_min < 0) or (f_max > 1):
        #x[i] = x[i].apply(lambda z: (z - f_mean)/f_std)
        #x_train[i] = x_train[i].apply(lambda z: (z - f_mean) / f_std)
        x[i] = (x[i] - f_min) / (f_max - f_min)
        x_train[i] = (x_train[i] - f_min) / (f_max - f_min)
        
        

x.to_csv('x_original.csv')
x_train.to_csv('x_train.csv')



# pandas is warning about indexing view vs copy so double check everything looks okay  
# this shows up when different types of data (int, float, category... ) are in same df
# since we're doing this column by column and columns must all be the same type 
# there should be no problems
print('scaled.............................') 
print(x.describe())
print(x_train.describe())





def convert_target(db, fileName):
    # convert target to one hot vector
    n_classes = len(db)
    y_classes = sorted(db['Class'].unique())
    classes = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '14', '15', '16']


    y_one_hot = pd.DataFrame(0, index=np.arange(len(db)), columns=classes)
    for c in classes:
        y_one_hot[c] = np.where(db['Class'] == int(c), 1, 0)

    y_one_hot.to_csv(fileName)   
    

convert_target(y, 'y_original.csv')
convert_target(y_train, 'y_train.csv')    
 






'''
correlations = data.corr().abs()['Class']
sorted_correlations = correlations.sort_values(ascending=False)
print(sorted_correlations)
'''





























