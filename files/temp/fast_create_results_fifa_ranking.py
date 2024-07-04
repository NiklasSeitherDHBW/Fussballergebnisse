import pandas as pd
import numpy as np
from joblib import Parallel, delayed
import multiprocessing
from tqdm import tqdm

df_fifa_ranking = pd.read_csv('files/input/fifa_ranking-2024-04-04.csv')
df_results = pd.read_csv('files/input/kaggle/results.csv')
df_goalscorers = pd.read_csv('files/input/kaggle/goalscorers.csv')
df_shootouts = pd.read_csv('files/input/kaggle/shootouts.csv')

# Spalte home_win befüllen
df_results["home_win"] = df_results.apply(
    lambda row: "draw" if row["home_score"] == row["away_score"] else "True" if row["home_score"] > row["away_score"] else "False",
    axis=1
)

fifa_ranking_cleaned = df_fifa_ranking.copy().dropna(subset=['rank'])

results_cleaned = df_results.copy().dropna(subset=['home_team', 'away_team'])

results_cleaned['home_score'] = results_cleaned['home_score'].fillna(results_cleaned.groupby('home_team')['home_score'].transform('mean'))
results_cleaned['away_score'] = results_cleaned['away_score'].fillna(results_cleaned.groupby('away_team')['away_score'].transform('mean'))

goalscorers_cleaned = df_goalscorers.dropna(subset=['scorer'])

goalscorers_cleaned['minute'] = goalscorers_cleaned['minute'].fillna(goalscorers_cleaned['minute'].mean())

# Add Fifa Ranking
latest_fifa_ranking = df_fifa_ranking.copy().sort_values(by='rank_date').drop_duplicates(subset=['country_full'], keep='last')

results = df_results.copy().merge(latest_fifa_ranking[['country_full', 'total_points']], left_on='home_team', right_on='country_full', how='left')
results = results.rename(columns={'total_points': 'home_team_strength'})
results = results.drop(columns=['country_full'])

results = results.copy().merge(latest_fifa_ranking[['country_full', 'total_points']], left_on='away_team', right_on='country_full', how='left')
results = results.rename(columns={'total_points': 'away_team_strength'})
results = results.drop(columns=['country_full'])

results["date"] = pd.to_datetime(results["date"])

def calculate_team_form(team, date, results_l, N=5):
    team_results = results_l[((results_l['home_team'] == team) | (results_l['away_team'] == team)) & (results_l['date'] < date)].sort_values(by='date').tail(N)
    if team_results.empty:
        return 0, 0
    home_games = team_results[team_results['home_team'] == team]
    away_games = team_results[team_results['away_team'] == team]
    
    avg_goals = (home_games['home_score'].sum() + away_games['away_score'].sum()) / N
    points = (home_games['home_score'] > home_games['away_score']).sum() * 3 + (home_games['home_score'] == home_games['away_score']).sum() + \
             (away_games['away_score'] > away_games['home_score']).sum() * 3 + (away_games['away_score'] == away_games['home_score']).sum()
    
    return avg_goals, points

def parallel_calculate_team_form(row, results_l):
    home_team_form_goals, home_team_form_points = calculate_team_form(row['home_team'], row['date'], results_l)
    away_team_form_goals, away_team_form_points = calculate_team_form(row['away_team'], row['date'], results_l)
    return home_team_form_goals, home_team_form_points, away_team_form_goals, away_team_form_points

def apply_form_calculation(df_chunk):
    df_chunk[["home_team_form_goals", "home_team_form_points", "away_team_form_goals", "away_team_form_points"]] = \
        df_chunk.apply(lambda row: parallel_calculate_team_form(row, results), axis=1, result_type='expand')
    return df_chunk

def parallel_apply(df, func, num_cores=multiprocessing.cpu_count()):
    df_split = np.array_split(df, num_cores)
    pool = Parallel(n_jobs=num_cores)
    
    results_l = []
    with tqdm(total=len(df)) as pbar:
        for result in pool(delayed(func)(chunk) for chunk in df_split):
            results_l.append(result)
            pbar.update(len(result))
            
    return pd.concat(results_l, axis=0)

results = parallel_apply(results, apply_form_calculation)
results['home_advantage'] = (~results['neutral']).astype(int)

print(results.head(5))
