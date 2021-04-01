import pandas as pd
from functions import convert_to_sscore
import numpy as np
import matplotlib.pyplot as plt

table_path = '/Users/ethan/OneDrive - University of Surrey/Coursework/Final Year/FYP/league_table.csv'
df = pd.read_csv(table_path, index_col='HE provider')

# Convert table to s scores
s_score = convert_to_sscore(df)

# Initialise weights
weights = np.random.rand(len(s_score.columns))

while True:
    try:
        uni = input('Enter a university for rank maximisation: ')
        if not uni in df.index:
            raise Exception
        break
    except Exception:
        print('That university is not in the table! Try again')


def run(weights):
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

    error = rank - 1
    return rank, score, error

def probability(change):
    prob = np.exp(-change * beta)
    return prob


highest_rank, highest_score, lowest_error = run(weights)
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
        if highest_rank == 1:
            break
        new_weights = weights.copy()
        new_weights[j] += ((2 * np.random.random()) - 1) * 0.1
        rank, score, error = run(new_weights)
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
    if highest_rank == 1:
        break
    beta += 0.01

print(f'Highest Rank: {highest_rank} --- Highest Score: {highest_score}')

if min(weights) < 0:
    weights = highest_weights + abs(min(highest_weights))

print(f'Weights are: {weights}')

plt.plot(errs)
plt.show()