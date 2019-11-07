import numpy as np
import pandas as pd
import sqlalchemy

SQL_CREDENTIALS = "root:1234"
engine = sqlalchemy.create_engine('mysql+pymysql://' + SQL_CREDENTIALS + '@localhost:3306/music_recommender_db')


def compute_listening_counts(list_of_users, n):
    set_of_users = np.random.choice(list_of_users, size=n, replace=False)
    print("%d users" % len(set_of_users))
    stmt = "SELECT user_id, track_id, COUNT(*) AS playcount FROM events WHERE user_id IN " + str(tuple(set_of_users)) + " GROUP BY user_id, track_id"
    playcounts_df = pd.read_sql(sql=stmt, con=engine)

    # Some users did not listen to any tracks
    diff = n - playcounts_df["user_id"].nunique()
    while diff > 0:
        print(diff)
        new_user_set = np.random.choice(list(set(list_of_users).difference(playcounts_df["user_id"].unique())), size=diff, replace=False)
        if diff > 1:
            stmt = "SELECT user_id, track_id, COUNT(*) AS playcount FROM events WHERE user_id IN " + str(tuple(new_user_set)) + " GROUP BY user_id, track_id"
        else:
            stmt = "SELECT user_id, track_id, COUNT(*) AS playcount FROM events WHERE user_id = " + str(new_user_set[0]) + " GROUP BY user_id, track_id"
        new_playcounts_df = pd.read_sql(sql=stmt, con=engine)
        playcounts_df = playcounts_df.append(new_playcounts_df)

        diff = n - playcounts_df["user_id"].nunique()

    return playcounts_df


def compute_le_split(events_df, test_size=0.01):
    testsize_df = events_df.groupby("user_id").size()
    print(testsize_df.head())

    trainset_df = pd.DataFrame()
    testset_df = pd.DataFrame()
    i = 1
    for user_id, group in events_df.groupby("user_id"):
        n_events = len(group)
        n_test = int(np.round(testsize_df.loc[user_id] * test_size))
        sorted_events_df = group.sort_values(by="timestamp", ascending=False)
        usertest_df = sorted_events_df.head(n_test)
        usertrain_df = sorted_events_df.tail(n_events - n_test)

        trainset_df = trainset_df.append(usertrain_df)
        testset_df = testset_df.append(usertest_df)

        print(i)
        i += 1

    return trainset_df, testset_df


    """events_df = pd.read_csv("../feature_engineering/data/lowms_les.csv", sep=";")
    n_events_df = events_df.groupby("user_id").size()
    playcounts_df = pd.read_csv("data/playcounts_track.csv", sep=";")

    train_df = pd.DataFrame()
    test_df = pd.DataFrame()

    i = 1
    for user_id, group in events_df.groupby("user_id"):
        n_test = int(np.floor(n_events_df.loc[user_id] * test_size))
        n_train = n_events_df.loc[user_id] - n_test
        #testset = group.sort_values(by="timestamp", ascending=False)["track_id"].head(n_test).values
        #trainset = group.sort_values(by="timestamp", ascending=False)["track_id"].tail(n_train).values

        df = group.sort_values(by="timestamp", ascending=False).groupby(by="track_id").head(1)
        df = df.merge(playcounts_df, left_on=["user_id", "track_id"], right_on=["user_id", "track_id"])
        df = df[["user_id", "track_id", "playcount"]]

        trainset = df.tail(n_train)
        testset = df.head(n_test)

        #new_row = pd.DataFrame(index=[user_id], data={"recommendations": [trainset.tolist()]})
        train_df = train_df.append(trainset)

        #new_row = pd.DataFrame(index=[user_id], data={"groundtruth": [testset.tolist()]})
        test_df = test_df.append(testset)

        print(i / 2074)
        i += 1

    print(test_df.head())

    return train_df, test_df"""
