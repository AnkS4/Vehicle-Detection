import pickle
import cv2
import numpy as np
import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from skimage.feature import hog
import time
from sklearn.preprocessing import StandardScaler
from moviepy.editor import VideoFileClip
from helper_functions import *
from scipy.ndimage.measurements import label

print('Reading pickle...')
car_pickle = pickle.load(open('car_pickle.p', 'rb'))
cspace = car_pickle['cspace']
conv = car_pickle['conv']
spatial_size = car_pickle['spatial_size']
hist_bins = car_pickle['hist_bins']
orient = car_pickle['orient']
pix_per_cell = car_pickle['pix_per_cell']
cell_per_block = car_pickle['cell_per_block']
svc = car_pickle['svc']
X_scalar = car_pickle['scaler']

path = './test_images/*'
test_images = glob.glob(path)

#xstart = 192
#xstop = 1279
ystart = 400 #390
ystops = [486, 502, 534, 646, 646] #[486, 502, 656, 656] 534|588
scales = [0.72, 1, 1.5, 2, 2.5]#[0.72, 1, 1.5, 2]
heat_threshold = 6 #3
window = 64

def find_cars(img, scale, window, conv, ystart, ystop):
    img_boxes = []
    draw_img = np.copy(img)
    heatmap = np.zeros_like(img[:, :, 0])
    img = img.astype(np.float32)/255
    count = 0

    img_roi = img[ystart:ystop, :, :]
    color_roi = convert_color(img_roi, conv=conv)

    if scale != 1:
        imshape = color_roi.shape
        color_roi = cv2.resize(color_roi, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

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


def process_image(img, heat_flag=False):
    labels = []
    heat_maps = np.zeros_like(img[:, :, 0])
    for j, ystop in zip(scales, ystops):
        out_img, heat_map = find_cars(img, j, window, conv, ystart, ystop)
        heat_maps += heat_map
    thresh_heat_map = apply_threshold(heat_maps, heat_threshold)
    labels = label(thresh_heat_map)
    draw_img = draw_labeled_bboxes(np.copy(img), labels)
    if heat_flag == False:
        return draw_img
    else:
        return draw_img, thresh_heat_map

for n, i in enumerate(test_images):
    print('Reading Image', n+1)
    img = mpimg.imread(i)
    labels = []
    heat_maps = np.zeros_like(img[:, :, 0])

    draw_img, thresh_heat_map = process_image(img, heat_flag=True)

    fig = plt.figure(figsize=(24, 24))
    visualize(fig, 1, 2, (draw_img, thresh_heat_map))
    #break

#Detect cars for test video
test_output = 'test_out.mp4'
clip = VideoFileClip('test_video.mp4')
test_clip = clip.fl_image(process_image)
test_clip.write_videofile(test_output, audio=False)

#Detect cars for project video
project_output = 'project_out.mp4'
clip = VideoFileClip('project_video.mp4')
test_clip = clip.fl_image(process_image)
test_clip.write_videofile(project_output, audio=False)