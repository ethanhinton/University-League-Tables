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


def run(weights, uni, maximise=True):
    rankings = s_score.copy()
    if min(weights) < 0:
        weights = weights + abs(min(weights))
    weights = weights / sum(weights)
    for ind, col in enumerate(rankings.columns):
        rankings[col] = rankings[col] * weights[ind]
    rankings['score'] = rankings.sum(axis=1)
    rankings['score'] = (rankings['score'] + abs(min(rankings['score'])))
    rankings['score'] = rankings['score'] / max(rankings['score'])
    rankings.sort_values(by='score', ascending=False, inplace=True)
    rankings['rank'] = [x for x in range(1, len(rankings)+1)]

    rank = rankings.loc[uni, 'rank']
    score = rankings.loc[uni, 'score']

    error = (rank - 1) if maximise else (len(rankings) - rank)

    return rank, score, error

def probability(change):
    prob = np.exp(-change * beta)
    return prob

num = 0
for uni in s_score.index:
    num += 1
    print(f'{uni} -- {num} of {len(df)}')
    for r in range(2):
        if r == 0:
            print('Maximising rank...')
        else:
            print('Minimising rank...')
        # Initialise weights
        weights = np.random.rand(len(s_score.columns))
        print(weights)

        highest_rank, highest_score, lowest_error = run(weights, uni) if r==0 else run(weights, uni, maximise=False)
        highest_weights = weights
        current_rank = highest_rank
        current_score = highest_score
        current_error = lowest_error
        beta = 1
        errs = [lowest_error]
        for i in range(1000):
            if i % 50 == 0:
                print(f'Current Highest Rank is {highest_rank} --- Highest Score is {highest_score}')
            for j in range(len(s_score.columns)):
                if r == 0:
                    if highest_rank == 1:
                        break
                else:
                    if highest_rank == len(s_score):
                        break
                
                new_weights = weights.copy()
                new_weights[j] += ((2 * np.random.random()) - 1) * 0.1
                rank, score, error = run(new_weights, uni) if r == 0 else run(new_weights, uni, maximise=False)
                if error <= current_error:
                    # print('Error is lower, adopting new weights')
                    current_error = error
                    current_score = score
                    current_rank = rank
                    weights = new_weights
                    if current_error < lowest_error:
                        highest_rank = current_rank
                        highest_score = current_score
                        lowest_error = current_error
                        highest_weights = weights
                elif np.random.random() < probability(error - current_error):
                    # print('Adopting higher error weights')
                    current_rank = rank
                    current_score = score
                    current_error = error
                    weights = new_weights
                # else:
                #     print('Error is higher, keeping weights')
            errs.append(current_error)
            if r == 0:
                if highest_rank == 1:
                    break
            else:
                if highest_rank == len(s_score):
                    break
            beta += 0.01

        print(f'Highest Rank: {highest_rank} --- Highest Score: {highest_score}')

        if min(weights) < 0:
            weights = highest_weights + abs(min(highest_weights))
        
        if r == 0:
            final_df.loc[uni, 'Max Rank'] = highest_rank
        else:
            final_df.loc[uni, 'Min Rank'] = highest_rank
        print(f'Weights are: {weights}')

final_df.to_csv('maximise_rank.csv')
