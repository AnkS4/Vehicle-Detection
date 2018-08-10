import cv2
import numpy as np
import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from skimage.feature import hog
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from helper_functions import *
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip

def find_cars(img, scale, window):
    img_boxes = []
    draw_img = np.copy(img)
    heatmap = np.zeros_like(img[:, :, 0])
    img = img.astype(np.float32)/255
    count = 0

    img_roi = img[ystart:ystop, :, :]
    color_roi = convert_color(img_roi, conv='RGB2YUV')

    if scale != 1:
        imshape = color_roi.shape
        color_roi = cv2.resize(color_roi, (np.int(imshape[1]/scale), np.int(imshape[1]/scale)))

    ch1 = color_roi[:, :, 0]
    ch2 = color_roi[:, :, 1]
    ch3 = color_roi[:, :, 2]    

    nxblocks = (ch1.shape[1]//pix_per_cell) - 1
    nyblocks = (ch1.shape[0]//pix_per_cell) - 1
    nfeat_per_block = orient*cell_per_block**2
    #window = 64
    nblocks_per_window = (window//pix_per_cell) - 1
    cells_per_step = 2 # Instead of overlap, define how many cells to stop
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    for xb in range(nxsteps):
        for yb in range(nysteps):
            count += 1
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step

            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            subimg = cv2.resize(color_roi[ytop:ytop+window, xleft:xleft+window], (window, window))#(64, 64))

            spatial_features = bin_spatial(subimg, size= spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)
            #print(spatial_features.shape)
            #print(hist_features.shape)
            #print(subimg.shape)

            test_features = X_scalar.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
            test_prediction = svc.predict(test_features)

            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                cv2.rectangle(draw_img, (xbox_left, ytop_draw+ystart), (xbox_left+win_draw, ytop_draw+win_draw+ystart),
                    (0, 0, 255))
                img_boxes.append(((xbox_left, ytop_draw+ystart), (xbox_left+win_draw, ytop_draw+win_draw)))
                heatmap[ytop_draw+ystart:ytop_draw+win_draw+ystart, xbox_left:xbox_left+win_draw] += 1
    return draw_img, heatmap


def process_image(img):
    out_img, heat_map = find_cars(img, scale)
    labels = label(heat_map)
    draw_img = draw_labeled_bboxes(np.copy(img), labels)
    return draw_img

cars = glob.glob('./vehicles/*/*.png')
notcars = glob.glob('./non-vehicles/*/*.png')
print(len(cars))
print(len(notcars))

cspace = 'YUV'
spatial_size = (4, 4) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 'ALL'
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off

samples = 3000
random_cars_i = np.random.randint(0, len(cars), samples)
random_cars = np.array(cars)[random_cars_i]
random_notcars_i = np.random.randint(0, len(notcars), samples)
random_notcars = np.array(notcars)[random_notcars_i]

t = time.time()
car_features = extract_features(random_cars, cspace, spatial_size, hist_bins, orient, pix_per_cell, cell_per_block,
    hog_channel, spatial_feat, hist_feat, hog_feat)
notcar_features = extract_features(random_notcars, cspace, spatial_size, hist_bins, orient, pix_per_cell, cell_per_block, 
    hog_channel, spatial_feat, hist_feat, hog_feat)

X = np.vstack((car_features, notcar_features)).astype(np.float64)
X_scalar = StandardScaler().fit(X)
scaled_X = X_scalar.transform(X)

y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

state = np.random.randint(0, 10)
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.1, random_state=state)

svc = LinearSVC()
t = time.time()
svc.fit(X_train, y_train)
print('Time required for training SVC:', round(time.time()-t, 2), 's')
print('Accuracy for SVC:', round(svc.score(X_test, y_test), 2))

path = './test_images/*'
test_images = glob.glob(path)

out_titles = []
ystart = 400
ystop = 700
scale = 1 #.5
scale2 = 1.5
heat_threshold = 5
window = 64
window2 = 128

for i in test_images:
    #t = time.time()
    img = mpimg.imread(i)
    out_img, heat_map = find_cars(img, scale, window)
    thresh_heat_map = apply_threshold(heat_map, heat_threshold)
    labels = label(thresh_heat_map)

    out_img2, heat_map2 = find_cars(img, scale, window2)
    thresh_heat_map2 = apply_threshold(heat_map2, heat_threshold)
    labels2 = label(thresh_heat_map2)

    thresh_heat_map_comb = thresh_heat_map | thresh_heat_map2

    #Draw bounding boxes on a copy of the image
    draw_img = draw_labeled_bboxes(img, labels)

    fig = plt.figure(figsize=(24, 24))
    visualize(fig, 1, 2, (draw_img, thresh_heat_map_comb), out_titles)

test_output = 'test_out.mp4'
clip = VideoFileClip('project_video.mp4')
test_clip = clip.fl_image(process_image)

test_clip.write_videofile(test_output, audio=False)