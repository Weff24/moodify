import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from song_features import GetSongFeatures


# danceability    energy    key    loudness    mode    speechiness    acousticness    instrumentalness    liveness    valence    tempo
# Removed tempo & loudness and accuracy went up
path = 'mood_features_dataset_v5.csv'
data = pd.read_csv(path)
dataframe = pd.DataFrame(data, columns=['seeds', 'energy', 'valence'])  # Can include danceability
sample = dataframe.sample(5000).to_numpy()
training_sample = sample[0:6000]
training_features = training_sample[:, 1:]
training_targets = training_sample[:, 0]

components = 3

# Gaussian Mixture Model
gm_model = GaussianMixture(n_components=components).fit(training_features)
gm_training_predictions = gm_model.predict(training_features)

gm_mean_to_mood = {}
for i in range(components):
    gm_group = training_sample[gm_training_predictions == i]
    if len(gm_group) != 0:
        group_targets = gm_group[:, 0]
        values, counts = np.unique(group_targets, return_counts=True)
        group_val = values[np.argmax(counts)]
        gm_mean_to_mood[i] = group_val

print(gm_mean_to_mood)


# Happy song test
# 'Happy' by Pharrel Williams
# 'My Songs Know What You Did In The Dark' by Fall Out Boy
# 'Ex's and Oh's' by Elle King
# 'See You Again' by Wiz Khalifa
# 'One Summer Day' by Kato
# 'Sad Song' by We The Kings
# 'This City' by Sam Fischer
# 'Surrender' by Natalie Taylor
spotify_ids = np.array(['60nZcImufyMA1MKQY3dcCH', '2E43WFS4rRc09za2r2GmZl', '70eDxAyAraNTiD6lx2ZEnH', '66CFbqJScx6zRieGllITcs', '7eo2HVkRxrDU3bQPwEiIBT', '05CrK6Q5VGtfPDtyQFJ4Kf', '3H3r2nKWa3Yk5gt8xgmsEt', '0ecC8p17cDNlxHXkuYqeR6'])

song_features = GetSongFeatures(spotify_ids)
predictions = gm_model.predict(song_features)
song_mood_predictions = np.empty(predictions.shape, dtype=object)
for i in range(len(predictions)):
    song_mood_predictions[i] = gm_mean_to_mood[predictions[i]]
print(song_mood_predictions)

