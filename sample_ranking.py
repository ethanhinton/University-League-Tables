import pandas as pd
from functions import convert_to_sscore
import numpy as np
import matplotlib.pyplot as plt


columns = ['% Satisfied with Teaching','% Satisfied with course','Continuation','Expenditure per student (fte)','Student: staff ratio','Career prospects','Value added score/10','Average Entry Tariff','% Satisfied with Assessment']
table_path = '/Users/ethan/OneDrive - University of Surrey/Coursework/Final Year/FYP/Data/Guardian/Guardian_University_Guide_2021.xlsx'
final_df = pd.read_excel(table_path, sheet_name='Institution', index_col='Name of Provider')
df = pd.read_excel(table_path, sheet_name='Institution', index_col='Name of Provider')[columns]

# Convert table to s scores
s_score = convert_to_sscore(df)
s_score['Student: staff ratio'] = -s_score['Student: staff ratio']

ranks = []
for i in range(500):
    weights = np.random.rand(9)
    temp = s_score.copy()
    for ind, weight in enumerate(weights):
        temp.iloc[:,ind] = s_score.iloc[:,ind] * weight
    temp['score'] = temp.sum(axis=1)
    temp['original rank'] = final_df['rank2021']
    temp.sort_values(by='score', inplace=True, ascending = False)
    temp['rank'] = [x for x in range(1, len(temp) + 1)]
    temp.sort_values(by='original rank', inplace=True)
    print(temp['score'])
    print(temp['rank'])
    ranks.append(list(temp['rank']))

max_ranks = ranks[0].copy()
min_ranks = ranks[0].copy()

print(min_ranks)
print('\n\n')
print(max_ranks)

for rank in ranks[1:]:
    for ind, elem in enumerate(rank):
        print(f'element is {elem}')
        print(f'min rank elem is {min_ranks[ind]}')
        print(f'max rank elem is {max_ranks[ind]}')
        if elem < min_ranks[ind]:
            print('replacing min rank element')
            min_ranks[ind] = elem
        elif elem > max_ranks[ind]:
            print('replacing max rank element')
            max_ranks[ind] = elem
        print(f'min rank elem is now {min_ranks[ind]}')
        print(f'max rank elem is now {max_ranks[ind]}')
        print('\n')


print(f'\n final ranks are \n\n{max_ranks}\n\n{min_ranks}')
final_df['max rank'] = max_ranks
final_df['min rank'] = min_ranks

final_df.to_csv('sample_rank.csv')

