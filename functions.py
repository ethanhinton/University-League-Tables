import pandas as pd
import numpy as np
from math import isnan
from difflib import SequenceMatcher as sm

class NaNError(Exception):
    pass

#Compares two similar tables and determines how different they are in a certain shared metric
#(e.g. how much two uni league tables differ in institutions rankings)
#NOTE: Make sure both tables have the same index column and the indexes match in both tables
def compare_df(table1, table2, metric):
    difference = []
    best = None
    worst = None
    for i in table1.index:
        try:
            val_tab1 = table1.loc[i, metric]
            val_tab2 = table2.loc[i, metric]
            if isnan(val_tab1):
                raise NaNError
            if isnan(val_tab2):
                raise NaNError
            change = (val_tab1 - val_tab2)/val_tab1 * 100
            difference.append(change)
            if change == min(difference):
                worst = i
            elif change == max(difference):
                best = i
        except KeyError:
            continue
        except NaNError:
            continue
    print('\n')
    return np.mean(np.abs(difference)), max(difference), abs(min(difference)), best, worst

def s_score(df,column):
    new = (df[column] - df[column].mean())/df[column].std()
    return new

def similar(name,main,second):
    insecond = []
    inmain = []
    if name in list(second.index):
        return str(name)
    for i in list(second.index):
        if name in i:
            insecond.append(i)
        elif i in name:
            inmain.append(i)
    if not insecond and not inmain:
        return None
    elif inmain:
        if len(inmain) > 1:
            sim = {}
            for elem in inmain:
                sim[elem] = sm(a=name,b=elem).ratio()
            return max(sim)
        else:
            return inmain[0]
    else:
        if len(insecond) > 1:
            sim = {}
            for elem in insecond:
                sim[elem] = sm(a=name,b=elem).ratio()
            return max(sim)
        else:
            return insecond[0]

# Looks for similar indexes in two tables and sets them to be the same index
# main = table to copy index from
# second = table to copy index to
def copy_index(main,second):
    if second.index.name == None:
        in_name = 'index'
    else:
        in_name = str(second.index.name)
    final = second.reset_index()
    for index, name in enumerate(second.index):
        new_index = similar(name,second,main)
        if new_index:
            final.loc[index,in_name] = new_index
    final.set_index(in_name, inplace=True)
    return final

def compare(df, uni1, uni2):
    df_new = df.copy()
    for col in df.columns:
        df_new[col] = s_score(df, col)
    x = df_new.loc[uni1]
    y = df_new.loc[uni2]
    return list(x), list(y)

def convert_to_sscore(df):
    df_new = df.drop(axis=1, labels='Subjects')
    for col in df_new.columns:
        x = s_score(df_new,col)
        df_new[col] = x

    df_new['Student/Staff Ratio'] = - df_new['Student/Staff Ratio']
    nonans = df_new.copy()
    for ind in df_new.index:
        nonans.loc[ind] = df_new.loc[ind].fillna(value=df_new.loc[ind].mean())
    return nonans

def offers_subjects(x, subjects):
    if set(subjects) - set(x):
        return False
    else:
        return True

def rank(df, weights):
    s_score = apply_weights(convert_to_sscore(df), weights)
    df['Score'] = s_score.sum(axis=1)
    df.sort_values(by='Score', ascending=False, inplace=True)
    df['Rank'] = [x for x in range(1, len(s_score)+1)]
    return df

def apply_weights(df, weights):
    for ind, col in enumerate(df.columns):
        df[col] = df[col] * (weights[ind] / sum(weights))
    return df
