import cv2
import numpy as np
import glob
from skimage.feature import hog
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from helper_functions import *

# Read in cars and notcars
images = glob.glob('./imgs/*.jpeg')
cars = []
notcars = []
for image in images:
    if 'image' in image or 'extra' in image:
        notcars.append(image)
    else:
        cars.append(image)

car = np.random.randint(0, len(cars))
notcar = np.random.randint(0, len(notcars))

car_image = mpimg.imread(cars[car])
notcar_image = mpimg.imread(notcars[notcar])

color_space = 'YUV'
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 1
spatial_size = (32, 32) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
y_start_stop = [480, 719] # Min and max in y to search in slide_window()

car_features, car_hog_image = single_img_features(car_image, color_space, spatial_size, hist_bins,
	orient, pix_per_cell, cell_per_block, hog_channel, spatial_feat, hist_feat, hog_feat, vis=True)
notcar_features, notcar_hog_image = single_img_features(notcar_image, color_space, spatial_size, hist_bins,
    orient, pix_per_cell, cell_per_block, hog_channel, spatial_feat, hist_feat, hog_feat, vis=True)

images = [car_image, car_hog_image, notcar_image, notcar_hog_image]
#titles = ['Car Image', 'Car HOG Image', 'Notcar Image', 'Notcar HOG Image']
fig = plt.figure(figsize=(12, 3))
visualize(fig, 1, 4, images)#, titles)