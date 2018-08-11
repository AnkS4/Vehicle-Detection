[//]: # (Image References)
[hog_HLS]: ./output_images/hog_HLS.png
[hog_HSV]: ./output_images/hog_HSV.png
[hog_RGB]: ./output_images/hog_RGB.png
[hog_YCrCb]: ./output_images/hog_YCrCb.png
[hog_YUV]: ./output_images/hog_YUV.png
[img1]: ./output_images/img1.png
[img2]: ./output_images/img2.png
[img3]: ./output_images/img3.png
[img4]: ./output_images/img4.png
[img5]: ./output_images/img5.png
[img6]: ./output_images/img6.png

# Vehicle-Detection
Detect vehicles in a video feed.

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.
---

### Code files description:

* [helper_functions.py](helper_functions.py) has functions defined to reduce the code of other files.
* [hog_test.py](hog_test.py) extracts hog, spatial, hist features & outputs the visualization of it.
---

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

#### 3. How I trained a classifier using selected features:

I started with extracting `3000` random car & noncar images for `spatial`, `color` and `hog` features. Then, I normalized the features and splitted them to train and test features with ratio of `0.9:0.1`. Then, I trained it with `LinearSVC`.

### Sliding Window Search

#### 1. How I implemented a sliding window search:

I decided to search the image for vehicle in `Y range of [400, 720]` and `complete X range`, because all of the vehicles appear in that region. Upper region in Y axis is filled with trees/skies. I started with window size of 64 for selected region.
I tried to reduce false positives by adding heat threshold of 5. 

Using parameters, `scale = 1, heat_threshold = 5, window = 64, samples=8500, colorspace='YUV'`. I got following image detection output:

![img1][img1]
![img2][img2]
![img3][img3]
![img4][img4]
![img5][img5]
![img6][img6]


#### 2. How I decided what scales to search and how much to overlap windows:

#### 3. Examples of test images to demonstrate how your pipeline is working:

#### 4. What did I do to optimize the performance of your classifier:


### Video Implementation

#### 1. Link to my final video output:


#### 2. How I implemented some kind of filter for false positives and some method for combining overlapping bounding boxes:

---
### Discussion

#### Problems/issues I faced in the implementation of this project:

