import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans


# danceability    energy    key    loudness    mode    speechiness    acousticness    instrumentalness    liveness    valence    tempo
# Removed tempo & loudness and accuracy went up
path = 'mood_features_dataset_v5.csv'
data = pd.read_csv(path)
dataframe = pd.DataFrame(data, columns=['seeds', 'energy', 'valence'])  # Can include danceability
sample = dataframe.sample(6000).to_numpy()
#sample[:, 3] = sample[:, 3] / -60    # Scale db to be within 0 and 1 like other values
#sample[:, -1] = sample[:, -1] / 250   # Scale tempo bpm to be within 0 and 1 like other values
#sample[:, -2] = sample[:, -2] * 5   # Scale valence by 5 to increase importance in classification
training_sample = sample[0:5000]
training_features = training_sample[:, 1:]
training_targets = training_sample[:, 0]
training_targets = training_targets.reshape(training_targets.shape[0], -1)
testing_sample = sample[5000:]
testing_features = testing_sample[:, 1:]
testing_targets = testing_sample[:, 0]
testing_targets = testing_targets.reshape(testing_targets.shape[0], -1)

for components in range(1, 21): # 8/9 components does well with just energy and valence. 10 does well with danceability too
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

    gm_testing_predictions = gm_model.predict(testing_features)
    gm_correct = 0
    for i in range(1000): # number of testing_points
        if gm_mean_to_mood[gm_testing_predictions[i]] in testing_targets[i]:
            gm_correct += 1
    gm_accuracy = gm_correct / 1000   
    ################ print('GM (' + str(components) + ' components): ' + str(gm_accuracy))

    # K-Means Model
    km_model = KMeans(n_clusters=components).fit(training_features)
    km_training_predictions = km_model.predict(training_features)

    km_mean_to_mood = {}
    for i in range(components):
        km_group = training_sample[km_training_predictions == i]
        if len(km_group) != 0:
            group_targets = km_group[:, 0]
            values, counts = np.unique(group_targets, return_counts=True)
            group_val = values[np.argmax(counts)]
            km_mean_to_mood[i] = group_val

    km_testing_predictions = km_model.predict(testing_features)
    km_correct = 0
    for i in range(1000): # number of testing_points
        if km_mean_to_mood[km_testing_predictions[i]] in testing_targets[i]:
            km_correct += 1
    km_accuracy = km_correct / 1000   
    ################ print('KM (' + str(components) + ' components): ' + str(km_accuracy))

    print('GM (' + str(components) + ' components): ' + str(gm_accuracy) + '    KM (' + str(components) + ' components): ' + str(km_accuracy))
    
    