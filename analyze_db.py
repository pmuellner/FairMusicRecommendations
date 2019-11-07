import pandas as pd
import numpy as np
import sqlalchemy
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pathlib
import os
import sys
import seaborn as sns
from collections import Counter
from functools import reduce

PARENT_DIRNAME = "db_analysis"

SQL_CREDENTIALS = "root:1234"
engine = sqlalchemy.create_engine('mysql+pymysql://' + SQL_CREDENTIALS + '@localhost:3306/music_recommender_db')
print(engine.table_names())

def drop_zeros(a):
    return [e for e in a if e != 0]

def plot_loglog(data, xlabel, ylabel):
    counter_dict = Counter(data)
    max_x = np.log10(max(counter_dict.keys()))
    max_y = np.log10(max(counter_dict.values()))
    max_base = max([max_x, max_y])

    keys = list(counter_dict.keys())
    min_x = np.log10(min(drop_zeros(keys)))

    bin_count = 50
    bins = np.logspace(min_x, max_base, num=bin_count)

    # Based off of: http://stackoverflow.com/questions/6163334/binning-data-in-python-with-scipy-numpy
    bin_means_y = (np.histogram(keys, bins, weights=list(counter_dict.values()))[0] /
                   np.histogram(keys, bins)[0])
    bin_means_x = (np.histogram(keys, bins, weights=keys)[0] /
                   np.histogram(keys, bins)[0])


    plt.xscale('log')
    plt.yscale('log')
    plt.plot(bin_means_x, bin_means_y)
    plt.show()

"""
acoustic features
"""
def acoustic_features():
    createAnalysisSubfolder(file_to_analyze='acoustic_features')
    stdout_orig = sys.stdout
    f = open(PARENT_DIRNAME + "/acoustic_features/log.txt", "w")
    sys.stdout = f

    acoustic_features_df = pd.read_sql(sql="acoustic_features", con=engine)
    numeric_columns = ["danceability", "energy", "loudness", "speechiness", "acousticness", "instrumentalness", "liveness",
                           "valence", "tempo"]
    df = acoustic_features_df[numeric_columns]
    df.columns = ["Danceability", "Energy", "Loudness", "Speechiness", "Acousticness", "Instrumentalness", "Liveness",
                           "Valence", "Tempo"]
    acoustic_features_df = df
    df.boxplot()
    plt.savefig(PARENT_DIRNAME + "/acoustic_features/box.png")
    plt.close()

    scaler = MinMaxScaler()
    acoustic_features_df = pd.DataFrame(index=acoustic_features_df.index,
                                        data=scaler.fit_transform(acoustic_features_df),
                                        columns=acoustic_features_df.columns)

    print("fdsa")
    acoustic_features_df.boxplot(vert=False, grid=False)
    plt.show()

    acoustic_features_df[numeric_columns].hist(sharex=True, sharey=True, grid=False)
    plt.tight_layout()
    plt.savefig(PARENT_DIRNAME + "/acoustic_features/hist.png", dpi=200)
    plt.close()

    print(acoustic_features_df.shape)
    print(acoustic_features_df.isna().sum())

    sys.stdout = stdout_orig
    f.close()

"""
albums
"""
def albums():
    statement = "SELECT artist_id FROM albums"
    albums_df = pd.read_sql(sql=statement, con=engine)
    print(albums_df.head())
    albums_df["artist_id"].value_counts().plot(kind="box")
    plt.savefig(PARENT_DIRNAME + "/albums/box.png")
    plt.close()

    print(albums_df.groupby("artist_id").size())
    print(albums_df.groupby("artist_id").size().sort_values(ascending=False))

    albums_df["artist_id"].value_counts().plot(kind="hist", label=" ")
    plt.savefig(PARENT_DIRNAME + "/albums/hist.png")
    plt.close()

    print(albums_df.shape)
    print(albums_df.isna().shape)



    hist, bins = np.histogram(albums_df.groupby("artist_id").size(), bins=100)
    bins = [b for b in bins if b > 0]
    print(bins)
    logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
    plt.scatter(logbins[1:], hist)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Albums")
    plt.ylabel("Artists")
    plt.grid(False)
    plt.show()


    count, bin_edges = np.histogram(albums_df["artist_id"], 100)
    n_albums, occurrences = zip(*sorted(zip(bin_edges[1:], count), key=lambda tup: tup[0])[::-1])
    plt.loglog(n_albums, occurrences)
    plt.xlabel("# Albums")
    plt.ylabel("Artist")
    plt.show()

"""
artist genres
"""
def artist_genres():
    createAnalysisSubfolder(file_to_analyze='artist_genres')
    stdout_orig = sys.stdout
    f = open(PARENT_DIRNAME + "/artist_genres/log.txt", "w")
    sys.stdout = f

    artist_genre_df = pd.read_sql(sql="artist_genre_map", con=engine)

    print(artist_genre_df.shape)
    n_genres_per_artist_df = artist_genre_df["mapping_dict"].apply(lambda s: s.count(":"))
    n_genres_per_artist_df.hist(grid=False)
    plt.savefig(PARENT_DIRNAME + "/artist_genres/hist.png", dpi=200)
    plt.close()

    sys.stdout = stdout_orig
    f.close()

"""
artists
"""
def artists():
    createAnalysisSubfolder(file_to_analyze='artists')
    stdout_orig = sys.stdout
    f = open(PARENT_DIRNAME + "/artists/log.txt", "w")
    sys.stdout = f

    artists_df = pd.read_sql(sql="artists", con=engine)
    print(artists_df.shape)

    sys.stdout = stdout_orig
    f.close()

    print(artists_df.head())



"""
events
"""
def events():
    """statement = "SELECT user_id, COUNT(*) FROM events GROUP BY user_id"
    df = pd.read_sql(sql=statement, con=engine)
    print(df)

    print(df["COUNT(*)"].describe())"""

    createAnalysisSubfolder(file_to_analyze='events')
    #stdout_orig = sys.stdout
    #f = open(PARENT_DIRNAME + "/events/log.txt", "w")
    #sys.stdout = f

    """statement = "SELECT COUNT(user_id) FROM events"
    n_events_df = pd.read_sql(sql=statement, con=engine)

    # 351469333 events
    print(n_events_df)"""

    """statement = "SELECT user_id, COUNT(*) FROM events GROUP BY user_id"
    n_events_per_user_df = pd.read_sql(sql=statement, con=engine)
    n_events_per_user_df.set_index("user_id", inplace=True)
    n_events_per_user_df.hist()
    plt.savefig(PARENT_DIRNAME + "/events/hist_events_per_user.png")
    plt.close()
    print(n_events_per_user_df.describe())"""

    """statement = "SELECT user_id, COUNT(*) AS LEs FROM events GROUP BY user_id"
    n_events_per_user_df = pd.read_sql(sql=statement, con=engine)
    #n_events_per_user_df = n_events_per_user_df[n_events_per_user_df["LEs"] < 30000]
    #n_events_per_user_df.set_index("user_id", inplace=True)
    #n_events_per_user_df.hist()
    #plt.show()


    hist, bins = np.histogram(n_events_per_user_df["LEs"], bins=100)
    bins = [b for b in bins if b > 0]
    logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
    plt.scatter(logbins[1:], hist)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Listening Events")
    plt.ylabel("Users")
    plt.grid(False)
    plt.show()"""

    """user_count, bin_edges = np.histogram(n_events_per_user_df["COUNT(*)"], 100)
    playcounts, occurrences = zip(*sorted(zip(bin_edges[1:], user_count), key=lambda tup: tup[0])[::-1])
    # playcounts, occurrences = bin_edges[1:], user_count
    plt.loglog(playcounts, occurrences)
    plt.xlabel("Listening Events")
    plt.ylabel("Users")
    plt.show()"""

    statement = "SELECT track_id, COUNT(*) FROM events GROUP BY track_id"
    n_events_per_track_df = pd.read_sql(sql=statement, con=engine)
    n_events_per_track_df.set_index("track_id", inplace=True)
    n_events_per_track_df.hist()
    plt.savefig(PARENT_DIRNAME + "/events/hist_events_per_track.png")
    plt.close()
    print(n_events_per_track_df.describe())

    hist, bins = np.histogram(n_events_per_track_df["COUNT(*)"], bins=100)
    bins = [b for b in bins if b > 0]
    logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
    plt.scatter(logbins[1:], hist)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Listening Events")
    plt.ylabel("Tracks")
    plt.grid(False)
    plt.show()


    """track_count, bin_edges = np.histogram(n_events_per_track_df["COUNT(*)"], 100)
    playcounts, occurrences = zip(*sorted(zip(bin_edges[1:], track_count), key=lambda tup: tup[0])[::-1])
    # playcounts, occurrences = bin_edges[1:], user_coun
    plt.loglog(playcounts, occurrences)
    plt.xlabel("Listening Events")
    plt.ylabel("Track")
    plt.show()"""

    statement = "SELECT artist_id, COUNT(*) FROM events GROUP BY artist_id"
    n_events_per_artist_df = pd.read_sql(sql=statement, con=engine)
    n_events_per_artist_df.set_index("artist_id", inplace=True)
    n_events_per_artist_df.hist()
    plt.savefig(PARENT_DIRNAME + "/events/hist_events_per_artist.png")
    plt.close()
    print(n_events_per_artist_df.describe())

    hist, bins = np.histogram(n_events_per_artist_df["COUNT(*)"], bins=100)
    bins = [b for b in bins if b > 0]
    logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
    plt.scatter(logbins[1:], hist)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Listening Events")
    plt.ylabel("Artists")
    plt.grid(False)
    plt.show()


    """artist_count, bin_edges = np.histogram(n_events_per_artist_df["COUNT(*)"], 100)
    playcounts, occurrences = zip(*sorted(zip(bin_edges[1:], artist_count), key=lambda tup: tup[0])[::-1])
    # playcounts, occurrences = bin_edges[1:], user_coun
    plt.loglog(playcounts, occurrences)
    plt.xlabel("Listening Events")
    plt.ylabel("Artist")
    plt.show()"""




    """statement = "SELECT album_id, COUNT(*) FROM events GROUP BY album_id"
    n_events_per_album_df = pd.read_sql(sql=statement, con=engine)
    n_events_per_album_df.set_index("album_id", inplace=True)
    n_events_per_album_df.hist()
    plt.savefig(PARENT_DIRNAME + "/events/hist_events_per_album.png")
    plt.close()
    print(n_events_per_album_df.describe())"""


    """statement = "SELECT COUNT('index'), DAYNAME(FROM_UNIXTIME(timestamp)) AS day_name from events GROUP BY day_name"
    n_events_per_weekday_df = pd.read_sql(sql=statement, con=engine)
    it = pd.read_sql(sql=statement, con=engine, chunksize=100)
    n_events_per_weekday_df = next(it)
    print(n_events_per_weekday_df.info())

    n_events_per_weekday_df.set_index("day_name", inplace=True)
    n_events_per_weekday_df.sort_index(inplace=True, ascending=True)
    print(n_events_per_weekday_df)

    n_events_per_weekday_df = n_events_per_weekday_df.reindex(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])

    n_events_per_weekday_df.plot(kind="barh", legend=False)
    plt.ylabel("")
    plt.xlabel("Listening Events")
    plt.tight_layout()
    plt.grid(False)
    plt.savefig(PARENT_DIRNAME + "/events/bar_weekday.png")
    plt.show()

    statement = "SELECT COUNT('index'), HOUR(FROM_UNIXTIME(timestamp)) AS h from events GROUP BY h"
    n_events_per_hour_df = pd.read_sql(sql=statement, con=engine)
    n_events_per_hour_df.set_index("h", inplace=True)
    n_events_per_hour_df.sort_index(inplace=True)
    print(n_events_per_hour_df)
    
    n_events_per_hour_df.plot(kind="bar", legend=False)
    plt.tight_layout()
    plt.ylabel("Listening Events")
    plt.xlabel("Hour")
    plt.grid(False)
    plt.show()"""
    #plt.savefig(PARENT_DIRNAME + "/events/bar_hours.png")
    #plt.close()

    #sys.stdout = stdout_orig
    #f.close()

"""
hofstede
"""
def hofstede():
    createAnalysisSubfolder(file_to_analyze='hofstede')
    stdout_orig = sys.stdout
    f = open(PARENT_DIRNAME + "/hofstede/log.txt", "w")
    sys.stdout = f

    hofstede_df = pd.read_sql(sql='hofstede', con=engine)

    for col in hofstede_df.columns.values:
        hofstede_df.loc[hofstede_df[col] == "\\N", col] = np.nan

    hofstede_df['power_distance'] = pd.to_numeric(hofstede_df['power_distance'])
    hofstede_df['individualism'] = pd.to_numeric(hofstede_df['individualism'])
    hofstede_df['masculinity'] = pd.to_numeric(hofstede_df['masculinity'])
    hofstede_df['uncertainty_avoidance'] = pd.to_numeric(hofstede_df['uncertainty_avoidance'])
    hofstede_df['long_term_orientation'] = pd.to_numeric(hofstede_df['long_term_orientation'])
    hofstede_df['indulgence'] = pd.to_numeric(hofstede_df['indulgence'])
    print(hofstede_df.describe())

    hofstede_df.boxplot(vert=False)
    #plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    plt.savefig(PARENT_DIRNAME + "/hofstede/box.png", dpi=200)
    plt.close()

    hofstede_df.hist(sharex=True, sharey=True, grid=False)
    plt.tight_layout()
    plt.savefig(PARENT_DIRNAME + "/hofstede/hist.png", dpi=200)
    plt.close()

    print(hofstede_df.shape)
    print(hofstede_df.isna().sum())

    """import seaborn as sb
    a = hofstede_df["power_distance"].dropna()
    sb.distplot(a, kde=False)
    plt.show()"""

    sys.stdout = stdout_orig
    f.close()

"""
users
"""
def users():
    #createAnalysisSubfolder(file_to_analyze='users')
    #stdout_orig = sys.stdout
    #f = open(PARENT_DIRNAME + "/users/log.txt", "w")
    #sys.stdout = f

    statement = "SELECT username, gender, age, ctr, playcount FROM rel_users"
    users_df = pd.read_sql(sql=statement, con=engine)
    users_df["ctr"] = users_df["ctr"].astype("category")

    print(users_df[users_df["age"] == -1].sum()["age"])
    print(users_df[users_df["age"] == -1].sum()["age"] / len(users_df))

    print(users_df[users_df["age"] != -1]["age"].describe())

    print("dsafasdfsadf")
    print(users_df.head())


    users_df[users_df["age"] != -1]["age"].hist(bins=100, grid=False)
    plt.xlabel("Age")
    plt.ylabel("Users")
    plt.grid(False)
    plt.show()

    hist, bins = np.histogram(users_df["playcount"], bins=100)
    bins = [b for b in bins if b > 0]
    logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
    #plt.plot(logbins, hist)
    plt.scatter(logbins, hist)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Listening Events")
    plt.ylabel("Users")
    plt.grid("False")
    plt.show()


    """user_count, bin_edges = np.histogram(users_df["playcount"], 100)
    playcounts, occurrences = zip(*sorted(zip(bin_edges[1:], user_count), key=lambda tup: tup[0])[::-1])
    plt.loglog(playcounts, occurrences)
    plt.xlabel("Listening Events")
    plt.ylabel("Users")
    plt.show()"""

    print(users_df["ctr"].isna().sum())

    print(users_df["ctr"].value_counts())
    print(users_df["ctr"].value_counts() / len(users_df))

    males = users_df[users_df["gender"] == "m"]
    females = users_df[users_df["gender"] == "f"]
    neutrals = users_df[users_df["gender"] == "n"]

    print(len(users_df) - len(males) - len(females) - len(neutrals))
    print((len(users_df) - len(males) - len(females) - len(neutrals)) / len(users_df))

    print("count M, F, N: %f, %f, %f" % (len(males), len(females), len(neutrals)))
    print("percentages M, F, N: %f, %f, %f" % (len(males) / len(users_df), len(females) / len(users_df), len(neutrals) / len(users_df)))
    print("percentage invalid age: %f" % (len(users_df[users_df["age"] == -1]) / len(users_df)))

    print(males.loc[:, ["age", "playcount"]].describe())
    print(females.loc[:, ["age", "playcount"]].describe())
    print(neutrals.loc[:, ["age", "playcount"]].describe())

    plt.hist(males["age"].values, label="M", bins=50)
    plt.hist(females["age"].values, label="F", bins=50)
    plt.hist(neutrals["age"].values, label="N", bins=50)
    plt.legend(loc="upper right")
    plt.xlabel("age")
    plt.savefig(PARENT_DIRNAME + "/users/hist_age.png")
    plt.close()


    plt.hist(males["playcount"].values, label="M", bins=50)
    plt.hist(females["playcount"].values, label="F", bins=50)
    plt.hist(neutrals["playcount"].values, label="N", bins=50)
    plt.legend(loc="upper right")
    plt.xlabel("playcount")
    plt.savefig(PARENT_DIRNAME + "/users/hist_playcount.png")
    plt.close()

    males["ctr"].value_counts().plot(kind="bar", label="M", color="blue")
    females["ctr"].value_counts().plot(kind="bar", label="F", color="orange")
    neutrals["ctr"].value_counts().plot(kind="bar", label="N", color="green")
    plt.legend(loc="upper right")
    plt.xlabel("ctr")
    plt.tight_layout()
    plt.savefig(PARENT_DIRNAME + "/users/hist_ctr.png", dpi=200)
    plt.close()



    sys.stdout = stdout_orig
    f.close()


"""
social ties
"""
def social_ties():
    createAnalysisSubfolder(file_to_analyze='social_ties')
    stdout_orig = sys.stdout
    f = open(PARENT_DIRNAME + "/social_ties/log.txt", "w")
    sys.stdout = f

    statement = "SELECT V1, V2, weight FROM social_ties"
    social_ties_df = pd.read_sql(sql=statement, con=engine)

    social_ties_df["weight"] = social_ties_df["weight"].astype("category")
    social_ties_df["weight"].value_counts().sort_index(ascending=True).plot(kind='bar')
    plt.xlabel("weight")
    plt.tight_layout()
    plt.savefig(PARENT_DIRNAME + "/social_ties/bar_weight.png")
    plt.close()

    social_ties_df["V1"] = social_ties_df["V1"].astype("category")
    social_ties_df["V2"] = social_ties_df["V2"].astype("category")

    print(social_ties_df.shape)

    print(social_ties_df.head())

    statement = "SELECT V1, COUNT(*) FROM social_ties GROUP BY V1"
    n_edges_V1V2 = pd.read_sql(sql=statement, con=engine)
    n_edges_V1V2.set_index("V1", inplace=True)
    print(n_edges_V1V2.head())

    statement = "SELECT V2, COUNT(*) FROM social_ties GROUP BY V2"
    n_edges_V2V1 = pd.read_sql(sql=statement, con=engine)
    n_edges_V2V1.set_index("V2", inplace=True)
    print(n_edges_V2V1.head())

    n_edges_V1V2.hist(grid=False)
    plt.xlabel("# edges V1-V2")
    plt.savefig(PARENT_DIRNAME + "/social_ties/hist_edges_V1V2.png", dpi=200)
    plt.close()

    n_edges_V2V1.hist(grid=False)
    plt.xlabel("# edges V2-V1")
    plt.savefig(PARENT_DIRNAME + "/social_ties/hist_edges_V2V1.png", dpi=200)
    plt.close()

    sys.stdout = stdout_orig
    f.close()


"""
tracks
"""
def tracks():
    createAnalysisSubfolder(file_to_analyze='tracks')
    stdout_orig = sys.stdout
    #f = open(PARENT_DIRNAME + "/tracks/log.txt", "w")
    #sys.stdout = f

    statement = "SELECT artist_id, COUNT(*) FROM tracks GROUP BY artist_id"
    n_songs_per_artist_df = pd.read_sql(sql=statement, con=engine)
    n_songs_per_artist_df.set_index("artist_id", inplace=True)
    print(n_songs_per_artist_df.head())
    print(n_songs_per_artist_df.sort_values("COUNT(*)", ascending=False).head())


    #hist, _ = np.histogram(n_songs_per_artist_df["COUNT(*)"], bins=100)
    #print(hist)

    #plt.plot(hist)
    #plt.title("# songs per artist")
    #plt.show()

    print(len(n_songs_per_artist_df))

    hist, bins = np.histogram(n_songs_per_artist_df["COUNT(*)"], bins=100)
    bins = [b for b in bins if b > 0]
    logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
    plt.scatter(logbins[1:], hist)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Tracks")
    plt.ylabel("Artists")
    plt.grid(False)
    plt.show()


    count, bin_edges = np.histogram(n_songs_per_artist_df["COUNT(*)"], 100)
    n_songs, occurrences = zip(*sorted(zip(bin_edges[1:], count), key=lambda tup: tup[0])[::-1])
    # playcounts, occurrences = bin_edges[1:], user_count
    plt.loglog(n_songs, occurrences)
    plt.xlabel("# Tracks")
    plt.ylabel("Artist")
    plt.show()

    n_songs_per_artist_df.hist(bins=100)
    plt.savefig(PARENT_DIRNAME + "/tracks/hist_songs_per_artist.png")
    plt.close()

    n_songs_per_artist_df.boxplot()
    plt.savefig(PARENT_DIRNAME + "/tracks/box_songs_per_artist.png")
    plt.close()


"""
user mainstreaminess
"""
def mainstreaminess():
    createAnalysisSubfolder(file_to_analyze='mainstreaminess')
    stdout_orig = sys.stdout
    f = open(PARENT_DIRNAME + "/mainstreaminess/log.txt", "w")
    sys.stdout = f

    statement = "SELECT M_global_R_APC FROM user_mainstreaminess"
    mainstreaminess_df = pd.read_sql(sql=statement, con=engine)



    mainstreaminess_df.hist(grid=False, bins=100)
    plt.xlabel("Mainstreaminess")
    plt.ylabel("Users")
    plt.title("")
    #plt.tight_layout()
    #plt.savefig(PARENT_DIRNAME + "/mainstreaminess/hist.png", dpi=200)
    #plt.close()
    plt.show()


    mainstreaminess_df.boxplot()
    plt.tight_layout()
    plt.savefig(PARENT_DIRNAME + "/mainstreaminess/box.png")
    plt.close()

    sys.stdout = stdout_orig
    f.close()

"""
world happiness
"""
def world_happiness():
    createAnalysisSubfolder(file_to_analyze='world_happiness')
    stdout_orig = sys.stdout
    f = open(PARENT_DIRNAME + "/world_happiness/log.txt", "w")
    sys.stdout = f

    world_happiness_df = pd.read_sql(sql="world_happiness", con=engine)
    print(world_happiness_df.isna().sum())
    print(len(world_happiness_df))
    print("00000000000000")

    world_happiness_df.set_index(["country", "year"], inplace=True)

    # take latest entry
    world_happiness_df = world_happiness_df.groupby(level='country').tail(1)
    world_happiness_df.to_csv("world_happiness_latest.csv", sep=";")

    print(world_happiness_df.isna().sum())

    scaler = MinMaxScaler()
    transformed_df = pd.DataFrame(scaler.fit_transform(world_happiness_df[["Life Ladder", "Log GDP per capita",
                                                                           "Healthy life expectancy at birth",
                                                                           "Democratic Quality", "Delivery Quality",
                                                                           "Generosity"]]))
    transformed_df.columns = ["Life Ladder", "Log GDP per capita", "Healthy life expectancy at birth",
                                         "Democratic Quality", "Delivery Quality", "Generosity"]

    normalized_scores = world_happiness_df[["Social support", "Freedom to make life choices",
                                            "Perceptions of corruption", "Positive affect", "Negative affect",
                                            "Confidence in national government"]].append(transformed_df)

    print(normalized_scores.isna().sum())
    print(len(normalized_scores))

    normalized_scores.boxplot(vert=False)
    plt.show()
    plt.savefig(PARENT_DIRNAME + "/world_happiness/box_norm.png")
    plt.close()

    normalized_scores.hist(grid=False, sharex=True, sharey=True)
    plt.tight_layout()
    plt.savefig(PARENT_DIRNAME + "/world_happiness/hist_norm.png", dpi=200)
    plt.close()


    correlation_mat = normalized_scores.corr()
    print(correlation_mat)
    sns.heatmap(correlation_mat, annot=True, vmax=1, vmin=-1)
    #plt.tight_layout()
    plt.show()

    sys.stdout = stdout_orig
    f.close()

def createAnalysisSubfolder(file_to_analyze=None):
    dirname = os.path.dirname(__file__) + '/' + PARENT_DIRNAME + '/' + file_to_analyze
    if not os.path.exists(dirname):
        os.makedirs(dirname)
        return True
    else:
        return False


if __name__ == "__main__":
    db_analysis_dirname = os.path.dirname(__file__) + '/' + PARENT_DIRNAME
    if not os.path.exists(db_analysis_dirname):
        os.makedirs(db_analysis_dirname)

    plt.style.use("seaborn")
    SMALL_SIZE = 15
    MEDIUM_SIZE = 18
    BIGGER_SIZE = 21

    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=MEDIUM_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    #plt.rc('figure', figsize=(20.0, 10.0))

    #acoustic_features()
    albums()
    #artist_genres()
    #artists()
    #events()
    #hofstede()
    #users()
    #social_ties()
    #tracks()
    #mainstreaminess()
    #world_happiness()





