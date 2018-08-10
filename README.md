[//]: # (Image References)
[hog_HLS]: ./output_images/hog_HLS.png
[hog_HSV]: ./output_images/hog_HSV.png
[hog_RGB]: ./output_images/hog_RGB.png
[hog_YCrCb]: ./output_images/hog_YCrCb.png
[hog_YUV]: ./output_images/hog_YUV.png

# Vehicle-Detection
Detect vehicles in a video feed.

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

### Code files description

* [helper_functions.py](helper_functions.py) has functions defined to reduce the code of other files.
* [hog_test.py](hog_test.py) extracts hog, spatial, hist features & outputs the visualization of it.

### Histogram of Oriented Gradients (HOG)

#### 1. HOG features extraction:

`single_img_features` function in `helper_fuctions.py` provides the code for the feature extraction.

I started by exploring different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here are the output images with HOG parameters of 'orient = 9', 'pix_per_cell = 8', 'cell_per_block = 2', 'hog_channel = 1', 'spatial_size = (4, 4)', 'hist_bins = 32':

'HSL' Colorspace:
![HOG Image][hog_HLS]
'HSV' Colorspace:
![HOG Image][hog_HSV]
'RGB' Colorspace:
![HOG Image][hog_RGB]
'YCrCb' Colorspace:
![HOG Image][hog_YCrCb]
'YUV' Colorspace:
![HOG Image][hog_YUV]

#### 2. Why I settled on the final choice of HOG parameters:

I tried various combinations of parameters and tried plotting histogram features for many random images for both car and non-car caterogy. Finally, the chosen parameters gave distinctictive results and better accuracy than other.

For colorspace, `YCrCb` and `YUV` were giving good results. So, I proceeded with `YUV` at first.
