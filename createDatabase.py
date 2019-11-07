import pandas as pd
import sqlalchemy
import numpy as np
import time

SQL_CREDENTIALS = "root:1234"
#SQL_CREDENTIALS = "admin:"

#engine = sqlalchemy.create_engine('mysql+pymysql://root:1234@localhost:3306/music_recommender_db')
#engine = sqlalchemy.create_engine('mysql+pymysql://' + SQL_CREDENTIALS + '@localhost:3306/musicrec_db')
engine = sqlalchemy.create_engine('mysql+pymysql://' + SQL_CREDENTIALS + '@localhost:3306/music_recommender_db')

"""
world_happiness_df = pd.read_csv("LFM-1b_cultural_acoustic_features/world_happiness_report_2018.tsv", sep="\t", encoding='utf8')
world_happiness_df.to_sql(name="world_happiness", con=engine, index=False, if_exists='replace')



artists_df = pd.read_csv("LFM-1b_artists.txt", sep="\t", header=None, index_col=0)
artists_df.columns = ["artist_name"]
artists_df.index.name = "artist_id"
artists_df.to_sql(name="artists", con=engine, index=True, if_exists='replace')


user_mainstreaminess_df = pd.read_csv("user_mainstreaminess.txt", sep="\t", index_col='user_id')
user_mainstreaminess_df.to_sql(name="user_mainstreaminess", con=engine, index=True, if_exists='replace')


track_artist_LE_df = pd.read_csv("LFM-1b_track_artist_LE_sorted.txt", sep=";", error_bad_lines=False)
track_artist_LE_df.columns = ["song_title", "artist", "les"]
#track_artist_LE_df.set_index("song_title", inplace=True)
track_artist_LE_df.to_sql(name="track_artist_les", con=engine, index=True, if_exists='replace')


social_ties_df = pd.read_csv("LFM-1b_social_ties.txt", sep="\t")
social_ties_df.columns = ["V1", "V2", "weight"]
social_ties_df.to_sql(name="social_ties", con=engine, index=True, if_exists='replace')"""

users_df = pd.read_csv("LFM-1b_cultural_acoustic_features/users.tsv", sep="\t", index_col='user_id').sort_index()
users_df.to_sql(name="users", con=engine, index=True, if_exists='replace')

"""
artist_genre_map_df = pd.read_csv("genres/artist_genre_mapping.txt", sep=",", header=None)
artist_genre_map_df.columns = ["artist", "mapping_dict"]
artist_genre_map_df.set_index("artist", inplace=True)
artist_genre_map_df.to_sql(name="artist_genre_map", con=engine, index=True, if_exists='replace')


acoustic_features_df = pd.read_csv("LFM-1b_cultural_acoustic_features/acoustic_features_lfm_id.tsv",
                                index_col="track_id", sep="\t").sort_index()
acoustic_features_df.to_sql(name="acoustic_features", con=engine, index=True, if_exists='replace', chunksize=1024)


hofstede_df = pd.read_csv("LFM-1b_cultural_acoustic_features/hofstede.tsv", index_col="no", sep="\t").sort_index()
hofstede_df.to_sql(name="hofstede", con=engine, index=True)

albums_df = pd.read_csv("LFM-1b_albums.txt", sep="\t", header=None, index_col=0)
albums_df.columns = ["album_name", "artist_id"]
albums_df.index.name = "album_id"
albums_df.to_sql(name="albums", con=engine, index=True, chunksize=1024, if_exists='append')



nicknames_df = pd.read_csv("LFM-1b_users.txt", sep='\t', header=None)
nicknames_df.columns = ["user_id", "username", "realname", "ctr", "age", "gender", "playcount", "registered_unixtime"]
nicknames_df = nicknames_df.set_index("user_id").sort_index()
nicknames_df.to_sql(name="user_w_nicknames", con=engine, index=True)



tracks_df = pd.read_csv("LFM-1b_tracks.txt", sep="\t", header=None)
tracks_df.columns = ["song_id", "song_name", "artist_id"]

tracks_df["artist_id"] = tracks_df["artist_id"].astype('uint32')
tracks_df["song_id"] = tracks_df["song_id"].astype('uint32')

tracks_df.set_index("song_id", inplace=True)

tracks_df.to_sql(name="tracks", con=engine, index=True, if_exists='append', chunksize=1024)


i = 0
s = time.time()
for chunk in pd.read_csv("LFM-1b_cultural_acoustic_features/events.tsv",
                         dtype={"user_id": "uint32", "artist_id": "uint32", "album_id": "uint32", "track_id": "uint32"},
                         sep="\t", chunksize=10**6):
    print("chunk %d" % i)
    print(chunk.memory_usage().sum())
    chunk.to_sql(name="events", con=engine, index=True, if_exists='append', chunksize=1024)
    i += 1
    print(time.time() - s)
    print()
    s = time.time()

"""
lfm_users_df = pd.read_csv("LFM-1b_users.txt", sep="\t", header=None)
lfm_users_df.columns = ["user_id", "username", "realname", "ctr", "age", "gender", "playcount", "registered_unixtime"]
lfm_users_df = lfm_users_df.set_index("user_id").sort_index()

rel_users_df = pd.read_csv("LFM-1b_cultural_acoustic_features/users.tsv", sep="\t", index_col='user_id').sort_index()

users_df = rel_users_df.merge(lfm_users_df, how='left', on='user_id')
users_df.drop(columns=["country", "age_y", "gender_y", "playcount_y", "registered_unixtime_y"], inplace=True)
users_df.columns = ["age", "gender", "playcount", "registered_unixtime", "username", "realname", "ctr"]
df = users_df[["username", "realname", "gender", "age", "ctr", "playcount", "registered_unixtime"]]
df = df.drop(columns=["realname"])

df.to_sql(name="rel_users", con=engine, index=True, if_exists='replace')
