##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in lines #23 through #40 of the file called helper.py.
I started by reading in all the vehicle and non-vehicle images.  I used "hog" function provided from skimage.feature module to extract the HOG features from the image. I then explored different color spaces and different skimage.hog() parameters (orientations, pixels_per_cell, and cells_per_block).I grabbed random images from each of the two classes and displayed them to get a feel for what the hog output looks like.
The code for this step is contained in the fifth code cell of the IPython notebook.



![alt text][image2]

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and tried with multiple options I chosen orient = 12,pix_per_cell = 16,cell_per_block =2.
Reason:
Since orientation takes values from 6 to 12, I want to use the maximum bin size to track down maximum variation
While experimenting pix_per_cell value, 16 gave a nice representation of image. 
I choose cell_per_block (2,2 )to get 4 block size of the image. Increasing this value, magnifies the  gradient line, and decreasing the value to 1 gives very small gradient line, which doesn't represent the image well. So, I chosen to 2,2 


####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The code for this step is contained in the ninth code cell of the IPython notebook.I used only HOG feature to detect the vehicle. I tried using color and spatial features however there were many false positives. So, I decided to use only HOG features for detecting an image. 

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The code for this step is contained in the eleventh code cell of the IPython notebook. I tried to create multiple windows by increasing the 18 pixels for iteration and increase the scale to 0.5. However, by this method doesn't work well which has multiple false positives. So, I decided to experiment with manual initialization of 8 search window with 1.0,1.5,2.0,3.5 scale respectively. The code for this step is contained in the fifteen code cell of the IPython notebook.

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

The code for this step is contained in the twelfth code cell of the IPython notebook. I optimize the model performance by excluding color and spatial features had more than 6000 features. In the current version, HOG features which has 1296 features. This helped model to perform much faster. 
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Please find the video 

####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

The code for this step is contained in the twenty first code cell of the IPython notebook.
I compute the Heat Map with the same method as in the course. Then I use thresholds to eliminate false positive detections and applied scipylabels to draw a bounding boxes around the labeled regions




---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I followed the lecture and was implementing the all the code from the lecture. I had no problems till I hit hog sub-sampling window search. After training the classifier I am got below error. And I couldn't figure out the reason. I knew this error dealt with dimension mismatch but not sure how to fix this issue. After 4 days of analyzing the code, I rewrote the extract_features function with small amendment to the existing code. Which includes color and spatial features. It failed because classifier trained on 2580 features wherein test feature had 6108 feature because including all 3 features in the model. 

There are some false positives that are detected. To make it more robust I’d like to implement YOLO /SSD to detect the vehicles. 
I was also facing a problem with false positives. I tried tuning multiple parameters but none worked. After searching in Google, I found that we need to store the tracking object from previous frame. So I implemented the class Vehicle_Detect.

Error: ValueError: operands could not be broadcast together with shapes (1,6108) (2580,) (1,6108)

