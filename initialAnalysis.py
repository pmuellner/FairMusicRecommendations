import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import json
from sklearn.preprocessing import MinMaxScaler
from pandas.io import sql
import sqlite3


pd.set_option('display.max_columns', 20)

"""
user_mainstreaminess
"""
def user_mainstreaminess():
    user_mainstreaminess_df = pd.read_csv("user_mainstreaminess.txt", sep="\t")
    print(user_mainstreaminess_df.head())

    print(user_mainstreaminess_df.describe())

    mainstreaminess_scores = user_mainstreaminess_df.drop(columns=['user_id', 'user_idx', 'country'])

    # nan values, M_global_R_ALC and M_country_R_ALC contain only nans
    print(len(mainstreaminess_scores))
    print(mainstreaminess_scores.isna().sum())

    mainstreaminess_scores = mainstreaminess_scores.drop(columns=["M_global_R_ALC", "M_country_R_ALC"])

    # 47 unique countries, some are extremely overrepresented
    print(user_mainstreaminess_df["country"].value_counts().describe())
    user_mainstreaminess_df["country"].value_counts().plot.box()
    plt.show()

    user_mainstreaminess_df["country"].value_counts().plot(kind="bar")
    plt.show()


    mainstreaminess_scores.hist(bins=50)
    plt.show()

    mainstreaminess_scores.boxplot()
    plt.show()


"""
LFM-1b_track_artist_LE_sorted
"""
def track_artist_LE():
    # TODO: handle ; within song name
    track_artist_LE_df = pd.read_csv("LFM-1b_track_artist_LE_sorted.txt", sep=";", error_bad_lines=False)
    track_artist_LE_df.columns = ["Song Title", "Artist", "LEs"]
    print(track_artist_LE_df.head())

    print(track_artist_LE_df.describe())
    #track_artist_LE_df["LEs"].value_counts().plot.box()
    #plt.show()
    track_artist_LE_df["LEs"].value_counts().hist()
    plt.xlabel("LEs per song")
    plt.show()

    # encoding errors (e.g. songnames in kyrilic letters have ##### as artist)
    events_per_artist_df = track_artist_LE_df.groupby(by="Artist").sum()
    print(events_per_artist_df.head())
    events_per_artist_df.hist()
    plt.xlabel("LEs per artist")
    plt.plot()

    songs_per_artist_df = track_artist_LE_df.groupby(by="Artist").count().sort_values(by="LEs", ascending=False).drop(columns=["Song Title"])
    songs_per_artist_df.head().plot.box()
    plt.xlabel("n songs per artist")
    plt.show()

"""
LFM-1b_social_ties
"""
def social_ties():
    social_ties_df = pd.read_csv("LFM-1b_social_ties.txt", sep="\t")
    social_ties_df.columns = ["V1", "V2", "weight"]
    print(social_ties_df.head())
    print(social_ties_df.describe())
    print(social_ties_df.isna().sum())

    # only 42 weight categories
    print(social_ties_df.nunique())
    social_ties_df["weight"].astype("category")

    #social_ties_df["weight"].hist(bins=45)
    print(social_ties_df["weight"].value_counts())

    social_ties_df["weight"].value_counts().sort_index(ascending=True).plot()
    plt.xlabel("weight")
    plt.ylabel("occurrences")
    plt.show()


"""
users.tsv
"""
def users():
    users_df = pd.read_csv("LFM-1b_cultural_acoustic_features/users.tsv", sep="\t")
    print(users_df.head())
    print(users_df.describe())
    print(users_df.isna().sum())

    # 12139 users did define -1 as age
    print(users_df[users_df["age"] == -1].sum()["age"])


    # 36506 men, 13937 women, 4663 neutral
    print(users_df["gender"].value_counts())

    males = users_df[users_df["gender"] == "m"]
    females = users_df[users_df["gender"] == "f"]
    neutrals = users_df[users_df["gender"] == "n"]

    print("percentages M, F, N, NDEF: %f, %f, %f" % (len(males) / len(users_df), len(females) / len(users_df), len(neutrals) / len(users_df)))
    print("percentage invalid age: %f" % (len(users_df[users_df["age"] == -1]) / len(users_df)))

    print(males.loc[:, ["age", "playcount"]].describe())
    print(females.loc[:, ["age", "playcount"]].describe())
    print(neutrals.loc[:, ["age", "playcount"]].describe())

    plt.hist(males["age"].values, label="M", bins=50)
    plt.hist(females["age"].values, label="F", bins=50)
    plt.hist(neutrals["age"].values, label="N", bins=50)
    plt.legend(loc="upper right")
    plt.show()


    plt.hist(males["playcount"].values, label="M", bins=50)
    plt.hist(females["playcount"].values, label="F", bins=50)
    plt.hist(neutrals["playcount"].values, label="N", bins=50)
    plt.legend(loc="upper right")
    plt.show()


"""
artist_genre_mapping
"""
def artist_genre_map():
    artist_genre_map_df = pd.read_csv("genres/artist_genre_mapping.txt", sep=",", header=None)
    artist_genre_map_df.columns = ["artist", "mapping_dict"]
    artist_genre_map_df.set_index("artist", inplace=True)
    print(artist_genre_map_df.head())

    print(artist_genre_map_df.dtypes)


    n_genres_df = artist_genre_map_df["mapping_dict"].apply(lambda s: s.count(":"))
    print(n_genres_df.head())

    n_genres_df.hist(bins=100)
    plt.ylabel("n artists")
    plt.xlabel("n genres")
    plt.show()

"""
artist_tags
"""
def artist_tags():
    # same as artist_genre_map
    pass


"""
acoustic_features
"""
def acoustic_features():
    acoustic_features = pd.read_csv("LFM-1b_cultural_acoustic_features/acoustic_features_lfm_id.tsv",
                                    index_col="track_id", sep="\t").sort_index()
    print(acoustic_features.head())

    print(acoustic_features.isna().sum())

    acoustic_features.dropna(inplace=True)


    print(acoustic_features.describe())

    acoustic_features[["danceability", "energy", "speechiness", "acousticness", "instrumentalness", "liveness",
                       "valence"]].boxplot()
    plt.show()

    acoustic_features[["speechiness", "liveness"]].hist()
    plt.show()

"""
hofstede
"""
def hofstede():
    hofstede_df = pd.read_csv("LFM-1b_cultural_acoustic_features/hofstede.tsv", index_col="no", sep="\t").sort_index()
    print(hofstede_df.head())

    for col in hofstede_df.columns.values:
        hofstede_df.loc[hofstede_df[col] == "\\N", col] = np.nan

    print(hofstede_df.isna().sum())


    hofstede_df['power_distance'] = pd.to_numeric(hofstede_df['power_distance'])
    hofstede_df['individualism'] = pd.to_numeric(hofstede_df['individualism'])
    hofstede_df['masculinity'] = pd.to_numeric(hofstede_df['masculinity'])
    hofstede_df['uncertainty_avoidance'] = pd.to_numeric(hofstede_df['uncertainty_avoidance'])
    hofstede_df['long_term_orientation'] = pd.to_numeric(hofstede_df['long_term_orientation'])
    hofstede_df['indulgence'] = pd.to_numeric(hofstede_df['indulgence'])
    print(hofstede_df.dtypes)



    print(hofstede_df.describe())

    hofstede_df.boxplot()
    plt.show()

"""
world_happiness
"""
def world_happiness():
    """world_happiness_df = pd.read_csv("LFM-1b_cultural_acoustic_features/world_happiness_report_2018.tsv", sep="\t",
                                  index_col=["country", "year"])"""

    world_happiness_df = pd.read_csv("LFM-1b_cultural_acoustic_features/world_happiness_report_2018.tsv", sep="\t")

    #print(world_happiness_df.head())

    print(world_happiness_df["country"].nunique())
    world_happiness_df["country"].value_counts().plot(kind="hist")
    plt.xlabel("n years")
    plt.ylabel("n countries")
    plt.show()

    print(world_happiness_df.isna().sum())


    print(world_happiness_df.describe())


    world_happiness_df[["Social support", "Freedom to make life choices", "Generosity", "Perceptions of corruption",
                        "Positive affect", "Negative affect", "Confidence in national government"]].boxplot()

    plt.show()

    world_happiness_df[["Social support", "Freedom to make life choices", "Generosity", "Perceptions of corruption",
                        "Positive affect", "Negative affect", "Confidence in national government"]].hist()
    plt.show()

    scaler = MinMaxScaler()
    transformed_df = pd.DataFrame(scaler.fit_transform(world_happiness_df[["Life Ladder", "Log GDP per capita", "Healthy life expectancy at birth",
                                     "Democratic Quality", "Delivery Quality"]]))
    transformed_df.columns = ["Life Ladder", "Log GDP per capita", "Healthy life expectancy at birth",
                                     "Democratic Quality", "Delivery Quality"]
    transformed_df.boxplot()
    plt.show()

    # generosity is not in [0, 1]


    world_happiness_df["Healthy life expectancy at birth"].hist(bins=50)
    plt.show()

    normalized_scores = world_happiness_df[["Social support", "Freedom to make life choices", "Generosity", "Perceptions of corruption",
                        "Positive affect", "Negative affect", "Confidence in national government"]].append(transformed_df)
    normalized_scores.boxplot()
    plt.show()



def albums():
    albums_df = pd.read_csv("LFM-1b_albums.txt", sep="\t", header=None, index_col=0)
    albums_df.columns = ["album_name", "artist_id"]
    albums_df.index.name = "album_id"
    print(albums_df.head())

def tracks():
    tracks_df = pd.read_csv("LFM-1b_tracks.txt", sep="\t", header=None, index_col=0)
    tracks_df.columns = ["song_name", "artist_id"]
    tracks_df.index.name = "song_id"
    print(tracks_df.head())

def artists():
    artists_df = pd.read_csv("LFM-1b_artists.txt", sep="\t", header=None, index_col=0)
    artists_df.columns = ["artist_name"]
    artists_df.index.name = "artist_id"
    print(artists_df.head())

    artists_df.to_csv("database/artists.csv", sep=",")


def nicknames():
    nicknames_df = pd.read_csv("LFM-1b_users.txt", sep='\t', header=None)
    nicknames_df.columns = ["user_id", "username", "realname", "ctr", "age", "gender", "playcount", "registered_unixtime"]
    nicknames_df = nicknames_df.set_index("user_id").sort_index()
    print(nicknames_df.head())





def create_connection(db_file):
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except sqlite3.Error as e:
        print(e)

    return None


if __name__ == "__main__":
    #user_mainstreaminess()
    #track_artist_LE()
    #social_ties()
    #users()
    #artist_genre_map()
    #artist_tags()
    #acoustic_features()
    #hofstede()
    #world_happiness()
    #albums()
    #tracks()
    #artists()
    nicknames()







