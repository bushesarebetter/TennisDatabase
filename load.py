from re import S
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime

class Surface():
    HARD = "Hard"
    CLAY = "Clay"
    GRASS = "Grass"
    CARPET = "Carpet"
    OVERALL = "Overall" # TODO: Add this to the surface column in the data
def build_database(csv, output_file=None, output_file2=None):
    data = pd.read_csv(csv)


    winners = data[['winner_id', 'winner_name', 'winner_hand', 'winner_ht', 'winner_ioc', 'winner_age', 'tourney_date', 'surface', 'winner_rank', 'winner_rank_points', 'loser_id', 'loser_hand']].copy()
    winners.columns = ['id', 'name', 'hand', 'height', 'country', 'age', 'date', 'surface', 'rank', 'points', 'opponent_id', 'opponent_hand']

    winners['won'] = 1

    losers = data[['loser_id', 'loser_name', 'loser_hand', 'loser_ht', 'loser_ioc', 'loser_age', 'tourney_date', 'surface', 'loser_rank', 'loser_rank_points', 'winner_id', 'winner_hand']].copy()
    losers.columns = ['id', 'name', 'hand', 'height', 'country', 'age', 'date', 'surface', 'rank', 'points', 'opponent_id', 'opponent_hand']

    losers['won'] = 0

    player_matches = pd.concat([winners, losers])
    player_matches['date'] = pd.to_datetime(player_matches['date'], format='%Y%m%d')

    player_matches = player_matches.sort_values(by='date')
    player_matches['opponent_hand'] = player_matches['hand'].apply(lambda x: 'R' if x == 'L' else 'L')
    latest_player_info = player_matches.groupby('id').last().reset_index()

    player_profiles = latest_player_info[['id', 'name', 'hand', 'height', 'country', 'age', 'rank', 'points']].copy()

    if output_file:
        player_matches.to_csv(output_file, index=False)
    if output_file2:
        player_profiles.to_csv(output_file2, index=False)
    return player_profiles, player_matches

def get_player_stats(player_id, player_matches):
    player_data = player_matches[player_matches['id'] == player_id]

    if player_data.empty:
        return None

    win_percentage = player_data['won'].mean() * 100

    surface_stats = {}
    for surface in player_data['surface'].unique():
        surface_matches = player_data[player_data['surface'] == surface]
        if not surface_matches.empty:
            surface_stats[surface] = {
                'matches': len(surface_matches),
                'win_percentage': surface_matches['won'].mean() * 100
            }

    recent_matches = player_data.sort_values('date', ascending=False).head(10)
    recent_form = recent_matches['won'].mean() * 100
    latest_info = player_data.sort_values(by='date', ascending=False).iloc[0]

    player_stats = {
        'id': player_id,
        'name': latest_info['name'],
        'hand': latest_info['hand'],
        'height': latest_info['height'],
        'country': latest_info['country'],
        'age': latest_info['age'],
        'rank': latest_info['rank'],
        'rank_points': latest_info['points'],
        'overall_win_percentage': win_percentage,
        'recent_form': recent_form,
        'surface_stats': surface_stats,
        'matches_played': len(player_data),
        'last_match_date': player_data['date'].max()
    }

    return player_stats

def calculate_surface_ratings(player_matches, output_file=None):
    data = pd.read_csv(player_matches)
    
    data['tourney_date'] = pd.to_datetime(data['tourney_date'], format='%Y%m%d')

    winner_ids = data['winner_id'].unique()
    loser_ids = data['loser_id'].unique()
    all_ids = np.union1d(winner_ids, loser_ids)

    all_surfaces = data['surface'].unique()
    all_surfaces = np.append(all_surfaces, Surface.OVERALL)

    player_ratings = {}
    player_last_played = {}


    for player_id in all_ids:
        player_ratings[player_id] = {surface: 1500 for surface in all_surfaces}
        player_last_played[player_id] = {surface: None for surface in all_surfaces}

    K = 32
    data = data.sort_values(by='tourney_date')
    for _, match in data.iterrows():
        winner_id = match['winner_id']
        loser_id = match['loser_id']
        surface = match['surface']
        match_date = match['tourney_date']

        winner_rating = player_ratings[winner_id][surface]
        loser_rating = player_ratings[loser_id][surface]

        winner_expected = 1 / (1 + 10 ** ((loser_rating - winner_rating) / 400))
        loser_expected = 1 - winner_expected

        player_ratings[winner_id][surface] = winner_rating + K * (1 - winner_expected)
        player_ratings[loser_id][surface] = loser_rating + K * (0 - loser_expected)

        player_last_played[winner_id][surface] = match_date
        player_last_played[loser_id][surface] = match_date

        winner_overall_rating = player_ratings[winner_id][Surface.OVERALL]
        loser_overall_rating = player_ratings[loser_id][Surface.OVERALL]

        winner_overall_expected = 1 / (1 + 10 ** ((loser_overall_rating - winner_overall_rating) / 400))
        loser_overall_expected = 1 - winner_overall_expected

        player_ratings[winner_id][Surface.OVERALL] = winner_overall_rating + K * (1 - winner_overall_expected)
        player_ratings[loser_id][Surface.OVERALL] = loser_overall_rating + K * (0 - loser_overall_expected)

        player_last_played[winner_id][Surface.OVERALL] = match_date
        player_last_played[loser_id][Surface.OVERALL] = match_date

    rating_rows = []
    for id, surfaces in player_ratings.items():
        for surface, rating in surfaces.items():
            if player_last_played[id][surface] is not None:
                last_date = player_last_played[id][surface]
                rating_rows.append({
                    'id': id, 
                    'surface': surface, 
                    'rating': rating, 
                    'last_updated': last_date.strftime("%Y-%m-%d")
                })

    ratings_df = pd.DataFrame(rating_rows)

    if output_file:
        ratings_df.to_csv(output_file, index=False)
    return ratings_df

def load_csvs(matches, player_ratings, player_profiles, h2h):
    matches = pd.read_csv(matches)
    player_ratings = pd.read_csv(player_ratings)
    player_profiles = pd.read_csv(player_profiles)
    h2h = pd.read_csv(h2h, index_col=0)
    return matches, player_ratings, player_profiles, h2h

def calculate_win_probability(player1_id, player2_id, surface, player_ratings):
    player1_rating = player_ratings[(player_ratings['id'] == player1_id) & (player_ratings['surface'] == surface)]['rating'].values[0]
    player2_rating = player_ratings[(player_ratings['id'] == player2_id) & (player_ratings['surface'] == surface)]['rating'].values[0]
    return 1 / (1 + 10 ** ((player2_rating - player1_rating) / 400))

def head_to_head(matches, profiles, output_file=None):
    df = pd.DataFrame(0, index=profiles['id'].astype(int), columns=profiles['id'].astype(int))
    np.fill_diagonal(df.values, -1)
    for _, match in matches.iterrows():
        winner_id = match['winner_id']
        loser_id = match['loser_id']
        df.at[winner_id, loser_id] += 1
    if output_file:
        df.to_csv(output_file)
    return df

def calculate_h2h(player1_id, player2_id, h2h):
    player2_id = str(player2_id)
    return h2h.at[player1_id, player2_id]
player_profiles, player_matches = build_database('matches.csv', 'player_matches.csv', 'player_profiles.csv')
#surfaces_ratings = calculate_surface_ratings('matches.csv', 'player_ratings.csv')
#head_to_head(matches, player_profiles, 'head_to_head.csv')

matches = pd.read_csv('matches.csv')
player_matches, player_ratings, player_profiles, h2h = load_csvs('player_matches.csv', 'player_ratings.csv', 'player_profiles.csv', 'head_to_head.csv')
probability = calculate_win_probability(104745, 104925, Surface.OVERALL, player_ratings)

h2h_Nadal_Djokovic = calculate_h2h(104925, 104745, h2h) # for some UNKNOWN reason this is now a string


print(probability)
print(h2h_Nadal_Djokovic)
