import pandas as pd
import numpy as np
from math import isnan
from difflib import SequenceMatcher as sm
import collections

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
            print(inmain)
            ind = int(input(f'Which from {inmain} is the correct index translation for {name}? '))
            if ind == 0:
                return None
            return inmain[ind-1]
        else:
            return inmain[0]
    else:
        if len(insecond) > 1:
            ind = int(input(f'Which from {insecond} is the correct index translation for {name}? '))
            if ind == 0:
                return None
            return insecond[ind-1]
        else:
            return insecond[0]

def in_list(List, element):
    try:
        return List.index(element)
    except ValueError:
        return None

# Looks for similar indexes in two tables and sets them to be the same index
# main = table to copy index from
# second = table to copy index to
def copy_index(main,second):
    changes = {}
    for name in second.index:
        new_index = similar(name,second,main)
        if new_index:
            changes[name] = new_index
        else:
            changes[name] = name

    # Sometimes when converting index there will be duplicate names, this code will ask the user which of the duplcates
    # they want to keep in the dataframe
    dups = [item for item in collections.Counter(list(changes.values())).items() if item[1] > 1]
    if dups:
        for item in dups:
            keys = [key for key, val in changes.items() if val == item[0]]
            ind = int(input(f'Multiple indexes with the same name {item[0]}, choose which one to keep {keys}'))
            if ind == 0:
                continue
            keys.remove(keys[ind-1])
            for key in keys:
                changes.pop(key, None) 
        return changes      
    return changes


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
    df_new['Applications to Acceptance (%)'] = - df_new['Applications to Acceptance (%)']
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
