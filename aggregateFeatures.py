import pandas as pd
import numpy as np
import sqlalchemy

SQL_CREDENTIALS = "root:1234"
engine = sqlalchemy.create_engine('mysql+pymysql://' + SQL_CREDENTIALS + '@localhost:3306/music_recommender_db')


def _chooseTopKGenres(df, K):
    df_ = pd.DataFrame()
    print(len(df))
    for uid, row in df.iterrows():
        topKgenres = list(row.sort_values(ascending=False).head(K).index)
        #df_ = df_.append(pd.DataFrame({"user_id": uid, "genres": topKgenres}))
        new_row = pd.DataFrame({"user_id": uid, "genres": [topKgenres]})
        df_ = df_.append(new_row)
        print(len(df_) / len(df))

    df_.set_index("user_id", inplace=True)
    return df_

def _useAllGenres(df):
    genres_df = df.div(df.sum(axis=1), axis=0)
    return genres_df


"""
get acoustic features for tracks
features_df: (user_id, track_id, f1, f2, ...)
"""
def _constructAcousticFeatures():
    users_df = pd.read_csv("data/low_main_users.txt", sep=",")
    users_df.set_index("user_id", inplace=True)
    users_df["novelty_artist_avg_year"] = users_df["novelty_artist_avg_year"].replace("?", np.nan).astype("float")

    acoustic_features_df = pd.read_csv("data/acoustic_features_lfm_id.tsv", sep="\t")
    acoustic_features_df.set_index("track_id", inplace=True)

    user_set = set(users_df.index)
    statement = "SELECT user_id, track_id FROM events WHERE user_id IN " + str(tuple(user_set))
    user_track_df = pd.read_sql(con=engine, sql=statement)

    user_track_df.drop_duplicates(inplace=True)

    features_df = user_track_df.join(acoustic_features_df, on="track_id")
    features_df.to_csv("data/user_all_tracks_features.csv", sep=";", index=False)


"""
aggregate track features per user
"""
def aggAcousticFeaturesPerUser():
    _constructAcousticFeatures()

    features_df = pd.read_csv("data/user_all_tracks_features.csv", sep=";")
    numeric_columns = ["danceability", "energy", "loudness", "speechiness", "acousticness", "instrumentalness", "liveness", "valence", "tempo"]
    #aggregations = ["min", "max", "mean", "median", "std"]
    aggregations = ["mean"]
    aggregate_df = features_df.groupby(by="user_id")[numeric_columns].agg(aggregations)
    aggregate_df["mode"] = features_df.groupby(by="user_id")["mode"].agg(["mean"])

    flattened_column_names = []
    for feature in numeric_columns:
        for agg in aggregations:
            name = feature + "_" + agg
            flattened_column_names.append(name)
    flattened_column_names.append("mode_mean")

    aggregate_df.columns = flattened_column_names
    aggregate_df.to_csv("data/aggregated.csv", sep=";")


"""
combine user features and track features
"""
def combineUserFTrackF():
    users_df = pd.read_csv("data/low_main_users.txt", sep=",", index_col="user_id")
    users_df["novelty_artist_avg_year"] = users_df["novelty_artist_avg_year"].replace("?", np.nan).astype("float")

    features_df = pd.read_csv("data/aggregated.csv", sep=";", index_col="user_id")
    combined_df = users_df.join(features_df)
    combined_df.to_csv("data/user_track_features.csv", sep=";", index=True)

"""
add genres
"""
def combineUTGenreF():
    df = pd.read_csv("data/user_track_features.csv", sep=";", index_col="user_id")

    # big genres
    genres_df = pd.read_csv("data/LFM-1b_UGP_weightedPC_allmusic.txt", sep="\t", index_col="user_id")

    # TODO introduce threshold?
    n_distinct_genres_df = (genres_df != 0).sum(axis=1)
    n_distinct_genres_df = n_distinct_genres_df.to_frame()
    n_distinct_genres_df.columns = ["n_dist_genres"]

    genres_df = _chooseTopKGenres(df=genres_df, K=5)
    #df = df.join(genres_df, lsuffix="user", rsuffix="genre")
    #df = df.join(n_distinct_genres_df, lsuffix="user", rsuffix="genre")
    df = df.join(genres_df)
    df = df.join(n_distinct_genres_df)

    df.to_csv("data/features_with_biggenres.csv", sep=";", index=True)

    # small genres
    genres_df = pd.read_csv("data/LFM-1b_UGP_weightedPC_freebase.txt", sep="\t", index_col="user_id")

    # TODO introduce threshold?
    n_distinct_genres_df = (genres_df != 0).sum(axis=1)
    n_distinct_genres_df = n_distinct_genres_df.to_frame()
    n_distinct_genres_df.columns = ["n_dist_genres"]

    genres_df = _chooseTopKGenres(df=genres_df, K=5)
    df = pd.read_csv("data/user_track_features.csv", sep=";", index_col="user_id")
    df = df.join(genres_df)
    df = df.join(n_distinct_genres_df)

    #df = df.join(genres_df, lsuffix="user", rsuffix="genre")
    df.to_csv("data/features_with_smallgenres.csv", sep=";", index=True)

"""
add hofstede
"""
def combineUTGHofstedeF():
    statement = " SELECT ctr, power_distance, individualism, masculinity, uncertainty_avoidance, long_term_orientation, indulgence FROM hofstede"
    hofstede_df = pd.read_sql(con=engine, sql=statement)

    df = pd.read_csv("data/features_with_biggenres.csv", sep=";")
    biggenres_w_hofstede_df = pd.merge(left=df, right=hofstede_df, left_on="country", right_on="ctr")
    biggenres_w_hofstede_df.set_index("user_id", inplace=True)
    del biggenres_w_hofstede_df["ctr"]

    biggenres_w_hofstede_df.to_csv("data/feat_bigg_hofst.csv", sep=";", index=True)


    df = pd.read_csv("data/features_with_smallgenres.csv", sep=";")
    smallgenres_w_hofstede_df = pd.merge(left=df, right=hofstede_df, left_on="country", right_on="ctr")

    del smallgenres_w_hofstede_df["ctr"]
    smallgenres_w_hofstede_df.set_index("user_id", inplace=True)
    smallgenres_w_hofstede_df.to_csv("data/feat_smallg_hofst.csv", sep=";", index=True)


"""
add world happiness
"""
def combineUTGHWorldHappinessF():
    ctr_to_country = pd.read_sql(con=engine, sql="SELECT ctr, country FROM hofstede")
    ctr_to_country.set_index("ctr", inplace=True)
    world_happiness_df = pd.read_sql(sql="world_happiness", con=engine)
    world_happiness_df.set_index(["country", "year"], inplace=True)

    # take latest entry
    world_happiness_df = world_happiness_df.groupby(level='country').tail(1)

    df = pd.read_csv("data/feat_bigg_hofst.csv", sep=";", index_col="user_id")
    country_per_user_df = pd.DataFrame(columns=["country"])
    for uid, row in df.iterrows():

        ctr = row["country"]
        country = ctr_to_country.loc[ctr]
        country_per_user_df.loc[uid] = country
    country_per_user_df.index.name = "user_id"
    country_per_user_df.reset_index(inplace=True)

    wh_per_user_df = pd.merge(country_per_user_df, world_happiness_df, left_on="country", right_on="country")
    wh_per_user_df.set_index("user_id", inplace=True)


    df = pd.merge(df, wh_per_user_df, left_index=True, right_index=True)
    del df["country_y"]
    df.rename(columns={"country_x": "country"}, inplace=True)
    df.to_csv("data/feat_bigg_hofst_wh.csv", sep=";", index=True)


    df = pd.read_csv("data/feat_smallg_hofst.csv", sep=";", index_col="user_id")
    country_per_user_df = pd.DataFrame(columns=["country"])
    for uid, row in df.iterrows():

        ctr = row["country"]
        country = ctr_to_country.loc[ctr]
        country_per_user_df.loc[uid] = country
    country_per_user_df.index.name = "user_id"
    country_per_user_df.reset_index(inplace=True)

    wh_per_user_df = pd.merge(country_per_user_df, world_happiness_df, left_on="country", right_on="country")
    wh_per_user_df.set_index("user_id", inplace=True)


    df = pd.merge(df, wh_per_user_df, left_index=True, right_index=True)
    del df["country_y"]
    df.rename(columns={"country_x": "country"}, inplace=True)
    df.to_csv("data/feat_smallg_hofst_wh.csv", sep=";", index=True)





if __name__ == "__main__":
    #aggAcousticFeaturesPerUser()
    #combineUserFTrackF()
    combineUTGenreF()
    combineUTGHofstedeF()
    combineUTGHWorldHappinessF()