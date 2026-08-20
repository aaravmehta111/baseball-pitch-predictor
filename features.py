import pandas as pd
# taking the big 119-column dataframe, and returning a smaller dataframe
#  containing only the columns I actually want to keep
def select_direct_features(df):
    columns_to_keep = [
        'balls', 'strikes', 'outs_when_up', 'inning',
        'on_1b', 'on_2b', 'on_3b',
        'stand', 'p_throws', 'bat_score_diff',
        'n_thruorder_pitcher', 'n_priorpa_thisgame_player_at_bat',
        'pitch_number', 'sz_top', 'sz_bot',
        'pitcher_days_since_prev_game', 'pitcher_days_until_next_game',
        'pitcher', 'fielder_2'
    ]
    
    new_df = df[columns_to_keep].copy()
    
    new_df['on_1b'] = new_df['on_1b'].notna()
    new_df['on_2b'] = new_df['on_2b'].notna()
    new_df['on_3b'] = new_df['on_3b'].notna()
    
    return new_df

def add_previous_pitch_features(df):
    df = df.copy()
    df = df.sort_values(['game_pk', 'at_bat_number', 'pitch_number'])
    
    df['prev_pitch_1'] = df.groupby(['game_pk', 'at_bat_number'])['pitch_type'].shift(1)
    df['prev_pitch_2'] = df.groupby(['game_pk', 'at_bat_number'])['pitch_type'].shift(2)
    df['prev_pitch_3'] = df.groupby(['game_pk', 'at_bat_number'])['pitch_type'].shift(3)

    return df    
    
    
def add_rolling_velocity(df, window=5):
    df = df.copy()
    df = df.sort_values(['game_pk', 'at_bat_number', 'pitch_number'])
    
    df['rolling_velocity'] = (
        df.groupby('pitcher')['release_speed']
        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    )
    
    return df

def add_rolling_pitch_mix(df):
    df = df.copy()
    df = df.sort_values(['game_pk', 'at_bat_number', 'pitch_number'])
    
    pitch_dummies = pd.get_dummies(df['pitch_type']).astype(float)
    
    for pitch in pitch_dummies.columns:
        shifted = pitch_dummies[pitch].groupby([df['pitcher'], df['game_pk']]).shift(1)
        cumulative_count = shifted.groupby([df['pitcher'], df['game_pk']]).cumsum()
        pitch_number_so_far = df.groupby(['pitcher', 'game_pk']).cumcount()
        df[f'pct_{pitch}_so_far'] = (cumulative_count / pitch_number_so_far).fillna(0)
    
    return df


def add_pitch_effectiveness(df):
    df = df.copy()
    df = df.sort_values(['game_pk', 'at_bat_number', 'pitch_number'])
    
    df['is_whiff'] = (df['description'] == 'swinging_strike').astype(float)
    
    pitch_types = df['pitch_type'].unique()
    
    for pitch in pitch_types:
        if pd.isna(pitch):
            continue
        
        is_this_pitch = (df['pitch_type'] == pitch).astype(float)
        whiff_this_pitch = df['is_whiff'] * is_this_pitch
        
        shifted_whiffs = whiff_this_pitch.groupby([df['pitcher'], df['game_pk']]).shift(1)
        shifted_counts = is_this_pitch.groupby([df['pitcher'], df['game_pk']]).shift(1)
        
        cum_whiffs = shifted_whiffs.groupby([df['pitcher'], df['game_pk']]).cumsum()
        cum_counts = shifted_counts.groupby([df['pitcher'], df['game_pk']]).cumsum()
        
        df[f'whiff_rate_{pitch}_so_far'] = (cum_whiffs / cum_counts).fillna(0)
    
    return df

def add_season_pitch_mix(df):
    df = df.copy()
    
    pitch_dummies = pd.get_dummies(df['pitch_type']).astype(float)
    
    for pitch in pitch_dummies.columns:
        season_pct = pitch_dummies[pitch].groupby(df['pitcher']).transform('mean')
        df[f'season_pct_{pitch}'] = season_pct
    
    return df

def add_pitcher_role(df, threshold=50):
    df = df.copy()
    
    pitches_per_game = df.groupby(['pitcher', 'game_pk']).size().reset_index(name='pitches_in_game')
    avg_pitches_per_game = pitches_per_game.groupby('pitcher')['pitches_in_game'].mean()
    
    df['avg_pitches_per_game'] = df['pitcher'].map(avg_pitches_per_game)
    df['is_starter'] = (df['avg_pitches_per_game'] >= threshold).astype(int)
    
    return df

def build_full_feature_set(df):
    df = add_previous_pitch_features(df)
    df = add_rolling_velocity(df)
    df = add_rolling_pitch_mix(df)
    df = add_pitch_effectiveness(df)
    df = add_season_pitch_mix(df)
    df = add_pitcher_role(df)
    return df

def select_final_features(df):
    direct_features = [
        'balls', 'strikes', 'outs_when_up', 'inning',
        'on_1b', 'on_2b', 'on_3b',
        'stand', 'p_throws', 'bat_score_diff',
        'n_thruorder_pitcher', 'n_priorpa_thisgame_player_at_bat',
        'pitch_number', 'sz_top', 'sz_bot',
        'pitcher_days_since_prev_game', 'pitcher_days_until_next_game',
        'pitcher', 'fielder_2'
    ]
    
    engineered_fixed = [
        'prev_pitch_1', 'prev_pitch_2', 'prev_pitch_3',
        'rolling_velocity', 'avg_pitches_per_game', 'is_starter'
    ]
    
    dynamic_features = [col for col in df.columns if col.startswith('pct_') or col.startswith('whiff_rate_') or col.startswith('season_pct_')]
    
    target = ['pitch_type']  # your label
    
    final_columns = direct_features + engineered_fixed + dynamic_features + target
    
    result = df[final_columns].copy()
    result['on_1b'] = result['on_1b'].notna()
    result['on_2b'] = result['on_2b'].notna()
    result['on_3b'] = result['on_3b'].notna()
    
    return result

def add_chronological_split(model_data, full_data, split_date):
    model_data = model_data.copy()
    model_data['game_date'] = full_data['game_date']
    
    train = model_data[model_data['game_date'] < split_date].drop(columns=['game_date'])
    test = model_data[model_data['game_date'] >= split_date].drop(columns=['game_date'])
    
    return train, test

def filter_to_regular_and_postseason(df):
    return df[df['game_type'].isin(['R', 'F', 'D', 'L', 'W'])].copy()