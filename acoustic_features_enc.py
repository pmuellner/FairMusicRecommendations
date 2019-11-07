import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

import sqlalchemy

SQL_CREDENTIALS = "root:1234"
engine = sqlalchemy.create_engine('mysql+pymysql://' + SQL_CREDENTIALS + '@localhost:3306/music_recommender_db')

# TODO only use lowms
statement = "SELECT track_id, acousticness, danceability, energy, instrumentalness, liveness, speechiness, tempo, valence FROM acoustic_features"
acoustic_features_df = pd.read_sql(sql=statement, con=engine)
acoustic_features_df.set_index("track_id", inplace=True)

scaler = StandardScaler()
acoustic_features_df = pd.DataFrame(scaler.fit_transform(acoustic_features_df), columns=acoustic_features_df.columns, index=acoustic_features_df.index)

imp = SimpleImputer(missing_values=np.nan, strategy='mean')
af_prep_df = pd.DataFrame(imp.fit_transform(acoustic_features_df), columns=acoustic_features_df.columns, index=acoustic_features_df.index)
print(af_prep_df)

#ks = np.arange(2, 10, 1)
ks = [8]
scores = []
models = []
for k in ks:
    gmm = GaussianMixture(n_components=k, init_params="kmeans", n_init=1)
    preds = gmm.fit(af_prep_df)

    score = gmm.bic(af_prep_df)
    scores.append(score)

    models.append(gmm)
    print(k)
    break

best_idx = np.argmin(scores)
best_k = ks[best_idx]
best_score = scores[best_idx]
best_model = models[best_idx]

print("best k: %f" % best_k)
print("best score: %f" % best_score)

probs = best_model.predict_proba(af_prep_df)
df = pd.DataFrame(data=probs, index=af_prep_df.index, columns=["C" + str(i) for i in range(best_k)])
print(df)
df.to_csv("data/acoustic_features_gmm_enc.csv", sep=";")


user_tracks_df = pd.read_csv("data/user_track.csv", sep=";", index_col="track_id", usecols=["track_id", "user_id"])
track_encodings_df = pd.read_csv("data/acoustic_features_gmm_enc.csv", sep=";", index_col="track_id")

user_tracks_enc_df = user_tracks_df.join(track_encodings_df)
user_tracks_enc_df.dropna(inplace=True)

user_encodings_df = user_tracks_enc_df.groupby(by="user_id").agg("mean")
user_encodings_df.to_csv("data/user_enc.csv", sep=";")


# acoustic features
statement = "SELECT "








