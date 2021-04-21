import pandas as pd
import streamlit as st
from functions import convert_to_sscore, offers_subjects, compare, rank
from ast import literal_eval
import altair as alt


st.set_page_config(layout="wide")
st.title('A University League Table App')
st.write('An interactive application to view and manipulate UK university league table data')

# MAIN TABLE

df = pd.read_csv('league_table.csv', index_col='HE provider', converters={'Subjects':literal_eval})

# Minimum Subjects filter
subjects_offered = st.sidebar.number_input('Minimum Number of Subjects Offered', max_value=20, min_value=0)
df = df.loc[df['Subjects'].map(len) >= subjects_offered]

# Specific Subject Areas Filter
all_subs = []
for subjects in df['Subjects']:
    for subject in subjects:
        all_subs.append(subject)
all_subs = list(set(all_subs))
all_subs.sort()
specific_subjects = st.sidebar.multiselect('Subject Filter', all_subs)


if specific_subjects:
    df = df.loc[df['Subjects'].apply(offers_subjects, subjects=specific_subjects)]

weights = [1 for x in range(len(df.columns) - 1)]

# Get weights from sliders
weights_expander = st.beta_expander('Change Weights')
with weights_expander:
    weights_columns = st.beta_columns(int((len(df.columns) - 1) / 2))
    for ind, col in enumerate(df.columns):
        if col == 'Subjects':
            continue
        weights[ind] = weights_columns[int(ind/2)].slider(f'{col} Weighting', max_value=10, step=1, value=1, key=str(ind))

# Rank the universities
df = rank(df, weights)

# Set columns to show on table
columns = ['Rank',
           '% Satisfied with Teaching',
           '% Satisfied with Course',
           '% Satisfied with Assessment',
           'Continuation %',
           '% Graduates in High Skilled Work',
           'Applications to Acceptance (%)',
           'Student/Staff Ratio',
           'Average Salary',
           'Academic Services Expenditure per Student',
           'Facilities Expenditure per Student']
df = df[columns]

st.dataframe(df)

st.header('Compare Two Universities')
uni_col1, uni_col2 = st.beta_columns(2)
uni1 = uni_col1.selectbox('University 1', df.index)
uni2 = uni_col2.selectbox('University 2', df.index)
stats1, stats2 = compare(df, uni1, uni2)

comp_cols = ['% Satisfied with Teaching',
           '% Satisfied with Course',
           '% Satisfied with Assessment',
           'Continuation %',
           '% Graduates in High Skilled Work',
           'Applications to Acceptance (%)',
           'Student/Staff Ratio',
           'Average Salary',
           'Academic Services Expenditure per Student',
           'Facilities Expenditure per Student']

comparison = pd.DataFrame([df.loc[uni1], df.loc[uni2]])[comp_cols]
comparison.reset_index(inplace=True)
chart_columns = st.beta_columns(2)
n = 0
for col in comparison.columns[1:]:
    if n % 2 == 0:
        c = alt.Chart(comparison).mark_bar().encode(
            alt.X(col),
            alt.Y('index', title='')
        )
        chart_columns[0].altair_chart(c, use_container_width=True)
    else:
        c = alt.Chart(comparison).mark_bar().encode(
            alt.X(col),
            alt.Y('index', title='')
        )
        chart_columns[1].altair_chart(c, use_container_width=True)
    n += 1