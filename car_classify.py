import glob
import numpy as np
from helper_functions import *
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
import time
import pickle

cars = glob.glob('./vehicles/*/*.png')
notcars = glob.glob('./non-vehicles/*/*.png')
print('Total car images:', len(cars))
print('Total non-car images:', len(notcars))

cspace = 'YCrCb' #'YUV'
conv = 'RGB2YCrCb' #'RGB2YUV'
spatial_size = (32, 32) #(4, 4) Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 'ALL'
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off

'''
samples = 4000
random_cars_i = np.random.randint(0, len(cars), samples)
random_cars = np.array(cars)[random_cars_i]
random_notcars_i = np.random.randint(0, len(notcars), samples)
random_notcars = np.array(notcars)[random_notcars_i]
'''
cars = np.array(cars)
notcars = np.array(notcars)
print('Extracting features...')
t = time.time()
car_features = extract_features(cars, cspace, spatial_size, hist_bins, orient, pix_per_cell, cell_per_block,
    hog_channel, spatial_feat, hist_feat, hog_feat)
notcar_features = extract_features(notcars, cspace, spatial_size, hist_bins, orient, pix_per_cell, cell_per_block, 
    hog_channel, spatial_feat, hist_feat, hog_feat)

print('Normalizing featurs...')
X = np.vstack((car_features, notcar_features)).astype(np.float64)
X_scalar = StandardScaler().fit(X)
scaled_X = X_scalar.transform(X)

y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

print('Training features...')
state = np.random.randint(0, 10)
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.1, random_state=state)

print('Fitting features...')
svc = LinearSVC()
t = time.time()
svc.fit(X_train, y_train)
print('Time required for training SVC:', round(time.time()-t, 2), 's')
print('Accuracy for SVC:', round(svc.score(X_test, y_test), 2))

car_pickle = {}
car_pickle['cspace'] = cspace
car_pickle['conv'] = conv
car_pickle['spatial_size'] = spatial_size
car_pickle['hist_bins'] = hist_bins
car_pickle['orient'] = orient
car_pickle['pix_per_cell'] = pix_per_cell
car_pickle['cell_per_block'] = cell_per_block
car_pickle['svc'] = svc
car_pickle['scaler'] = X_scalar

pickle.dump(car_pickle, open('./car_pickle.p', 'wb'))
print('Car Clasification pickle saved.')