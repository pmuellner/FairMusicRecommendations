import pandas as pd
import numpy as np
import sqlalchemy
from collections import Counter
import ast
import matplotlib.pyplot as plt

SQL_CREDENTIALS = "root:1234"
engine = sqlalchemy.create_engine('mysql+pymysql://' + SQL_CREDENTIALS + '@localhost:3306/music_recommender_db')

TEST_CONFIG = " LIMIT 100"

def get_listeningevents_on_weekend(list_of_users):
    statement = "SELECT user_id, COUNT(*) AS LEs FROM events WHERE user_id in " + str(tuple(list_of_users)) + \
                " AND DAYNAME(FROM_UNIXTIME(timestamp)) in " + str(("Saturday", "Sunday")) + " GROUP BY user_id"
    print(statement)
    weekend_df = pd.read_sql(sql=statement, con=engine)
    weekend_df.set_index("user_id", inplace=True)

    statement = "SELECT user_id, COUNT(*) AS LEs FROM events WHERE user_id in " + str(tuple(list_of_users)) + \
                " AND DAYNAME(FROM_UNIXTIME(timestamp)) in " + str(("Monday", "Tuesday", "Wednesday", "Thursday", "Friday")) + " GROUP BY user_id"
    print(statement)
    weekday_df = pd.read_sql(sql=statement, con=engine)
    weekday_df.set_index("user_id", inplace=True)

    return weekend_df / (weekend_df + weekday_df)

def get_listeningevents_workday(list_of_users):
    print(len(list_of_users))
    list_of_hours = list(range(7, 19))
    print(list_of_hours)
    statement = "SELECT user_id, COUNT(*) AS LEs FROM events WHERE user_id IN " + str(tuple(list_of_users)) + \
                " AND HOUR(FROM_UNIXTIME(timestamp)) IN " + str(tuple(list_of_hours)) + \
                " AND DAYNAME(FROM_UNIXTIME(timestamp)) in " + str(("Monday", "Tuesday", "Wednesday", "Thursday", "Friday")) + " GROUP BY user_id"
    workday_df = pd.read_sql(sql=statement, con=engine)
    workday_df.set_index("user_id", inplace=True)


    list_of_hours = list(range(19, 24)) + list(range(0, 8))
    print(list_of_hours)
    statement = "SELECT user_id, COUNT(*) AS LEs FROM events WHERE user_id IN " + str(tuple(list_of_users)) + \
                " AND HOUR(FROM_UNIXTIME(timestamp)) IN " + str(tuple(list_of_hours)) + \
                " AND DAYNAME(FROM_UNIXTIME(timestamp)) in " + str(("Saturday", "Sunday")) + \
                " GROUP BY user_id"
    non_workday_df = pd.read_sql(sql=statement, con=engine)
    non_workday_df.set_index("user_id", inplace=True)


    return workday_df / (workday_df + non_workday_df)

def get_listeningevents_in_evening(list_of_users):
    list_of_hours = list(range(17, 24)) + [0]
    statement = "SELECT user_id, COUNT(*) AS LEs FROM events WHERE user_id IN " + str(tuple(list_of_users)) + \
                " AND HOUR(FROM_UNIXTIME(timestamp)) IN " + str(tuple(list_of_hours)) + " GROUP BY user_id"
    print(statement)
    evening_df = pd.read_sql(sql=statement, con=engine)
    evening_df.set_index("user_id", inplace=True)

    list_of_hours = list(range(1, 17))
    statement = "SELECT user_id, COUNT(*) AS LEs FROM events WHERE user_id IN " + str(tuple(list_of_users)) + \
                " AND HOUR(FROM_UNIXTIME(timestamp)) IN " + str(tuple(list_of_hours)) + " GROUP BY user_id"
    print(statement)
    workday_df = pd.read_sql(sql=statement, con=engine)
    workday_df.set_index("user_id", inplace=True)

    return evening_df / (workday_df + evening_df)


def get_bursting_behaviour(list_of_users):
    list_of_hours = list(range(6, 24)) + [0]
    statement = "SELECT user_id, HOUR(FROM_UNIXTIME(timestamp)) AS FULL_HOUR FROM events WHERE user_id IN " + \
                str(tuple(list_of_users)) + " AND HOUR(FROM_UNIXTIME(timestamp)) IN " + str(tuple(list_of_hours))
    daytime_df = pd.read_sql(sql=statement, con=engine).set_index("user_id")
    print(daytime_df)

    statement = "SELECT user_id, DATEDIFF(MAX(FROM_UNIXTIME(timestamp)), MIN(FROM_UNIXTIME(timestamp))) AS N_DAYS " \
                "FROM events where user_id IN " + str(tuple(list_of_users)) + " GROUP BY user_id"
    n_days_active_df = pd.read_sql(sql=statement, con=engine).set_index("user_id")
    print(n_days_active_df)

    # TODO normalize over days active or n listeningevents?
    les_per_hour_df = daytime_df.groupby(by="user_id")["FULL_HOUR"].value_counts().to_frame()
    les_per_hour_df.columns = ["count"]
    print(les_per_hour_df)
    les_per_hour_df["count"] = les_per_hour_df["count"] / n_days_active_df["N_DAYS"]
    print(les_per_hour_df)

    std_per_hour_df = les_per_hour_df.groupby(by="user_id").std()
    print(std_per_hour_df)

    std_per_hour_df.hist(bins=30)
    plt.show()


    return std_per_hour_df


def get_longest_genre_session(list_of_users):
    events_df = pd.read_csv("data/lowms_les.csv", sep=";")

    track_genres_df = pd.read_csv("data/track_genres.csv", sep=";", header=None)
    track_genres_df.columns = ["track_id", "genres"]
    track_genres_df.set_index("track_id", inplace=True)

    events_genres_df = events_df.merge(track_genres_df, left_on="track_id", right_index=True)
    events_genres_df = events_genres_df[events_genres_df["genres"] != "[]"]
    events_genres_df.set_index("user_id", inplace=True)

    longest_subsequences_df = pd.DataFrame()
    for user_id in events_genres_df.index.unique():
        group = pd.DataFrame(events_genres_df.loc[[user_id]]).sort_values(by=["timestamp"], ascending=True)
        subset = set()
        len_subsequence = 0
        len_longest_subsequence = 0
        first_in_subsequence = True
        for _, row in group.iterrows():
            current_genres = set(ast.literal_eval(row["genres"]))
            if first_in_subsequence:
                subset = current_genres
                first_in_subsequence = False
            else:
                subset = subset.intersection(set(current_genres))

            if len(subset) == 0:
                if len_subsequence > len_longest_subsequence:
                    len_longest_subsequence = len_subsequence

                len_subsequence = 0
            else:
                len_subsequence += 1

        longest_subsequences_df = longest_subsequences_df.append(pd.DataFrame({"user_id": [user_id], "len_longest_sseq": [len_longest_subsequence]}))
        print(len(longest_subsequences_df) / len(list_of_users), len_longest_subsequence)

    longest_subsequences_df.set_index("user_id", inplace=True)

    return longest_subsequences_df

def get_average_session_length(list_of_users):
    events_df = pd.read_csv("data/lowms_les.csv", sep=";")

    track_genres_df = pd.read_csv("data/track_genres.csv", sep=";", header=None)
    track_genres_df.columns = ["track_id", "genres"]
    track_genres_df.set_index("track_id", inplace=True)

    events_genres_df = events_df.merge(track_genres_df, left_on="track_id", right_index=True)
    events_genres_df = events_genres_df[events_genres_df["genres"] != "[]"]
    events_genres_df.set_index("user_id", inplace=True)

    session_durations_df = pd.DataFrame()
    for user_id in events_genres_df.index.unique():
        session_lengths = []
        group = pd.DataFrame(events_genres_df.loc[[user_id]]).sort_values(by=["timestamp"], ascending=True)
        subset = set()
        len_subsequence = 0
        first_in_subsequence = True
        for _, row in group.iterrows():
            current_genres = set(ast.literal_eval(row["genres"]))
            if first_in_subsequence:
                subset = current_genres
                first_in_subsequence = False
            else:
                subset = subset.intersection(set(current_genres))

            if len(subset) == 0:
                session_lengths.append(len_subsequence)
                len_subsequence = 0

                first_in_subsequence = True
            else:
                len_subsequence += 1

        # TODO alternative to mean, median? Maybe take only longest sessions (e.g. Q3)
        if len(session_lengths) > 0:
            session_durations_df = session_durations_df.append(pd.DataFrame({"user_id": [user_id], "avg. session duration": [np.mean(session_lengths)]}))
        else:
            session_durations_df = session_durations_df.append(pd.DataFrame({"user_id": [user_id], "avg. session duration": [0.0]}))
        #session_durations_df = session_durations_df.append(pd.DataFrame({"user_id": [user_id], "avg. session duration": [np.median(session_lengths)]}))
        print(len(session_durations_df) / len(list_of_users))

    session_durations_df.set_index("user_id", inplace=True)

    print(session_durations_df)

    return session_durations_df



if __name__ == "__main__":
    users_df = pd.read_csv("data/low_main_users.txt", sep=",", squeeze=True, usecols=["user_id"])
    #rel_events_on_weekend_df = get_listeningevents_on_weekend(list_of_users=users_df.values.tolist())
    #rel_events_on_weekend_df.to_csv("data/rel_events_on_weekend.csv", sep=";", index=True)

    rel_events_workday_df = get_listeningevents_workday(list_of_users=users_df.values.tolist())
    rel_events_workday_df.to_csv("data/rel_events_within_workday.csv", sep=";", index=True)

    #rel_events_in_evening_df = get_listeningevents_in_evening(list_of_users=users_df.values.tolist())
    #rel_events_in_evening_df.to_csv("data/rel_events_in_evening.csv", sep=";", index=True)

    #std_of_users_df = get_bursting_behaviour(list_of_users=users_df.values.tolist())
    #std_of_users_df.to_csv("data/std_of_users.csv", sep=";")

    #longest_genre_session_df = get_longest_genre_session(list_of_users=users_df.values.tolist())
    #longest_genre_session_df.to_csv("data/longest_genres_session.csv", sep=";")

    #session_lengths_df = get_average_session_length(list_of_users=users_df.values.tolist())
    #session_lengths_df.to_csv("data/session_lengths.csv", sep=";")

    rel_events_on_weekend_df = pd.read_csv("data/rel_events_on_weekend.csv", sep=";", index_col="user_id")
    rel_events_in_evening_df = pd.read_csv("data/rel_events_in_evening.csv", sep=";", index_col="user_id")
    bursting_df = pd.read_csv("data/std_of_users.csv", sep=";", index_col="user_id")
    genre_sessions_df = pd.read_csv("data/longest_genres_session.csv", sep=";", index_col="user_id")
    session_lengths_df = pd.read_csv("data/session_lengths.csv", sep=";", index_col="user_id")

    plt.hist(rel_events_on_weekend_df["LEs"].values, label="weekend", bins=30, alpha=0.7)
    plt.legend()
    plt.show()

    plt.hist(rel_events_in_evening_df["LEs"].values, label="evening", bins=30, alpha=0.7)
    plt.legend()
    plt.show()

    plt.hist(bursting_df["count"].values, label="std", bins=30, alpha=0.7)
    plt.legend()
    plt.show()

    plt.hist(genre_sessions_df["len_longest_sseq"].values, label="longest sequence", bins=30, alpha=0.7)
    plt.legend()
    plt.show()

    plt.hist(session_lengths_df["avg. session duration"].values, label="avg. session duration", bins=30, alpha=0.7)
    plt.legend()
    plt.show()

    """merged_df = rel_events_on_weekend_df.copy()
    merged_df = merged_df.merge(rel_events_in_evening_df, left_index=True, right_index=True, suffixes=["_weekend", "_evening"])

    merged_df.boxplot()
    plt.show()"""

    full_df = rel_events_on_weekend_df.copy()
    full_df = full_df.merge(rel_events_in_evening_df, left_index=True, right_index=True)
    full_df = full_df.merge(session_lengths_df, left_index=True, right_index=True)

    full_df.columns = ["weekend", "evening", "avg. session length"]
    full_df.to_csv("data/all_temporal_features.csv", sep=";")




