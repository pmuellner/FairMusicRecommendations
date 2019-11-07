import pandas as pd
import numpy as np
import sqlalchemy
from collections import Counter

COL_SEPARATOR = "_123456789987654321_"
N_TOP_GENRES = 10

SQL_CREDENTIALS = "root:1234"
engine = sqlalchemy.create_engine('mysql+pymysql://' + SQL_CREDENTIALS + '@localhost:3306/music_recommender_db')



def _artist_name_to_id():
    statement = "SELECT * FROM artists"
    df = pd.read_sql(sql=statement, con=engine)
    df.columns = ["artist_id", "artist_name"]
    df.set_index("artist_name", inplace=True)

    return df

def _artist_id_to_name():
    statement = "SELECT * FROM artists"
    df = pd.read_sql(sql=statement, con=engine)
    df.columns = ["artist_id", "artist_name"]
    df.set_index("artist_id", inplace=True)

    return df

def convert_artist_name_to_id(name):
    artists_df = _artist_name_to_id()
    if name in artists_df.index:
        return artists_df.loc[name]
    else:
        return np.nan

def create_user_to_track():
    users_df = pd.read_csv("data/low_main_users.txt", sep=",")
    users_df.set_index("user_id", inplace=True)

    print(len(users_df))

    rel_tracks = pd.read_csv("data/relevant_tracks_lowms.txt", header=None)
    print(len(rel_tracks))
    rel_tracks = rel_tracks[0].values

    user_set = set(users_df.index)
    """statement = "SELECT user_id, track_id, artist_id FROM events WHERE user_id IN " + str(tuple(user_set))
    user_track_df = pd.read_sql(con=engine, sql=statement)
    user_track_df.drop_duplicates(inplace=True)
    user_track_df = user_track_df[user_track_df["track_id"].isin(rel_tracks)]
    user_track_df.to_csv("data/user_track.csv", sep=";", index=False)"""

    statement_events = "SELECT user_id, track_id, artist_id, timestamp FROM events WHERE user_id IN " + str(tuple(user_set))
    #les_df = pd.read_csv("data/low_main_users.txt", sep=",")
    #les_df.set_index("user_id", inplace=True)
    les_df = pd.read_sql(con=engine, sql=statement_events)
    print(len(les_df))
    print(les_df["user_id"].nunique())


    les_df = les_df[les_df["track_id"].isin(rel_tracks)]
    les_df.to_csv("data/lowms_les.csv", sep=";", index=False)

def _genres_to_list(string_of_genres):
    if string_of_genres == "NaN":
        return np.nan
    else:
        return string_of_genres.split("\t")[:-1]


def prepare_spotify_microgenres(tracks=True, artists=True):
    if tracks:
        f = open("data/LFM-1b_spotify_microgenre_annotations/LFM-1b_artist_track_genres_spotify.txt", mode='r', encoding='utf8', newline='\r\n')
        f_new = open("data/artist_track_genres_prepared.csv", mode="w+", encoding="utf8")
        is_header = True
        i = 0
        for line in f:
            if is_header:
                is_header = False
                continue
            else:
                new_row = line.replace("\t", COL_SEPARATOR, 5)
                f_new.write(new_row)
            i += 1

            print(i / 566757)
        f_new.close()

    if artists:
        f = open("data/LFM-1b_spotify_microgenre_annotations/LFM-1b_artist_genres_spotify.txt", mode='r', encoding='utf8', newline='\r\n')
        f_new = open("data/artist_genres_prepared.csv", mode="w+", encoding="utf8")
        is_header = True
        i = 0
        for line in f:
            if is_header:
                is_header = False
                continue
            else:
                new_row = line.replace("\t", COL_SEPARATOR, 1)
                f_new.write(new_row)
            i += 1

            print(i / 585051)
        f_new.close()


"""
preprocessing
"""
#create_user_to_track()

prepare_spotify_microgenres(tracks=False, artists=True)

user_track_df = pd.read_csv("data/user_track.csv", sep=";", index_col="user_id")
del user_track_df["artist_id"]


rel_tracks = pd.read_csv("data/relevant_tracks_lowms.txt", header=None)
rel_tracks = rel_tracks[0].values
print(len(rel_tracks))
"""
track genres
"""
artist_track_genres_df = pd.read_csv("data/artist_track_genres_prepared.csv", sep=COL_SEPARATOR, header=None)
artist_track_genres_df.columns = ["track_id", "track_name", "artist_id", "artist_name", "listening_events", "genres"]
artist_track_genres_df.drop_duplicates(subset=["track_id", "track_name"], inplace=True)
artist_track_genres_df.set_index("track_id", inplace=True)


artist_track_genres_df = artist_track_genres_df.loc[rel_tracks].dropna()
print(len(artist_track_genres_df))

artist_track_genres_df["genres"] = artist_track_genres_df.apply(lambda d: _genres_to_list(str(d[-1])), axis=1)
artist_track_genres_df["genres"].to_csv("data/track_genres.csv", sep=";", index=True)

exit()


"""
artist genres
"""
artist_genres_df = pd.read_csv("data/artist_genres_prepared.csv", sep=COL_SEPARATOR, header=None)
artist_genres_df.columns = ["artist_name", "genres"]
artists_df = _artist_name_to_id()
artist_genres_df = artist_genres_df.merge(artists_df, left_on="artist_name", right_index=True)
artist_genres_df.drop_duplicates(subset=["artist_id", "artist_name"], inplace=True)
artist_genres_df.set_index("artist_id", inplace=True)

#events_df = pd.read_csv("data/lowms_les.csv", sep=";")
#rel_artists = list(set(events_df["artist_id"]))
#artist_track_genres_df = artist_track_genres_df.loc[rel_artists].dropna()


artist_genres_df["genres"] = artist_genres_df.apply(lambda d: _genres_to_list(str(d[-1])), axis=1)

"""
aggregation
"""
merged_df = user_track_df.merge(artist_track_genres_df, left_on="track_id", right_index=True)
track_genres_df = merged_df.groupby("user_id")["genres"].agg("sum").to_frame()
track_genres_df.rename(columns={"genres": "track_genres"}, inplace=True)

del merged_df["genres"]
merged_df = merged_df.merge(artist_genres_df, left_on="artist_id", right_index=True)
merged_df.rename(columns={"artist_name_x": "artist_name", "genres": "artist_genres"}, inplace=True)
del merged_df["artist_name_y"]
artist_genres_df = merged_df.groupby("user_id")["artist_genres"].agg("sum").to_frame()


"""print("n track genres")
print(len(set(track_genres_df["track_genres"].sum())))
print("n artist genres")
print(len(set(artist_genres_df["artist_genres"].sum())))"""

unique_track_genres = set(track_genres_df["track_genres"].sum())

with open('unique_spotify_microgenres.txt', 'w') as f:
    for item in unique_track_genres:
        f.write("%s\n" % item)





"""
choose top k genres for user (1. track, 2. artist)
"""
genres_per_user_df = track_genres_df.join(artist_genres_df)
topK_genres_per_user_df = pd.DataFrame()

print(genres_per_user_df.head())

for uid, row in genres_per_user_df.iterrows():
    if not type(row["track_genres"]) is type(list()) or not type(row["artist_genres"]) is type(list()):
        continue
    track_genres_occ = Counter(row["track_genres"])
    artist_genres_occ = Counter(row["artist_genres"])

    # Delete top 5 genres
    #top5_genres = ["rock", "pop", "electronic", "alternativerock", "metal"]
    top5_genres = []

    track_genres_occ = dict(track_genres_occ)
    artist_genres_occ = dict(artist_genres_occ)

    for genre in top5_genres:
        if genre in track_genres_occ:
            del track_genres_occ[genre]
        if genre in artist_genres_occ:
            del artist_genres_occ[genre]

    sorted_track_genres, sorted_track_genres_occ = [genre for genre, occ in sorted(track_genres_occ.items(), key=lambda tup: tup[1], reverse=True)]
    sorted_artist_genres, sorted_artist_genres_occ = [genre for genre, occ in sorted(artist_genres_occ.items(), key=lambda tup: tup[1], reverse=True)]

    print("=== sorted track genres ===")
    print(sorted_track_genres)

    n_track_genres = len(sorted_track_genres)
    n_artist_genres = len(sorted_artist_genres)

    if n_track_genres >= N_TOP_GENRES:
        new_row = pd.DataFrame(data={"user_id": uid, "top_genres": [sorted_track_genres[:N_TOP_GENRES]]})
        topK_genres_per_user_df = topK_genres_per_user_df.append(new_row)
    elif n_artist_genres >= N_TOP_GENRES:
        new_row = pd.DataFrame(data={"user_id": uid, "top_genres": [sorted_artist_genres[:N_TOP_GENRES]]})
        topK_genres_per_user_df = topK_genres_per_user_df.append(new_row)
    else:
        # TODO: top overall genres?
        # not necessary
        pass

    print(len(topK_genres_per_user_df))

topK_genres_per_user_df.set_index("user_id", inplace=True)
topK_genres_per_user_df.to_csv("data/top_genres_per_user.csv", sep=";", index=True)




"""
genres_per_user_df = track_genres_df.join(artist_genres_df)
artist_genres_per_user = pd.DataFrame()
track_genres_per_user = pd.DataFrame()
for uid, row in genres_per_user_df.iterrows():
    track_genres_occ = Counter(row["track_genres"])
    artist_genres_occ = Counter(row["artist_genres"])

    sorted_track_genres = [genre for genre, _ in sorted(track_genres_occ.items(), key=lambda tup: tup[1], reverse=True)]
    sorted_artist_genres = [genre for genre, _ in sorted(artist_genres_occ.items(), key=lambda tup: tup[1], reverse=True)]

    n_track_genres = len(sorted_track_genres)
    n_artist_genres = len(sorted_artist_genres)

    new_row = pd.DataFrame(data={"user_id": uid, "top_genres": [set(sorted_track_genres)]})
    track_genres_per_user = track_genres_per_user.append(new_row)

    new_row = pd.DataFrame(data={"user_id": uid, "top_genres": [set(sorted_artist_genres)]})
    artist_genres_per_user = artist_genres_per_user.append(new_row)


track_genres_per_user.set_index("user_id", inplace=True)
track_genres_per_user.to_csv("data/all_track_genres_per_user.csv", sep=";", index=True)

artist_genres_per_user.set_index("user_id", inplace=True)
artist_genres_per_user.to_csv("data/all_artist_genres_per_user.csv", sep=";", index=True)
"""

