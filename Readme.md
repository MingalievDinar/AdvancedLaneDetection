## Writeup Template

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./examples/undistort_example.png "Road Transformed"
[image3]: ./examples/binary_example_1.png "Binary Example1"
[image7]: ./examples/binary_example_2.png "Binary Example2"
[image8]: ./examples/binary_example_3.png "Binary Example3"
[image4]: ./examples/pers_trans.png "Warp Example"
[image9]: ./examples/pers_trans2.png "Warp Example"
[image5]: ./examples/poly.png "Fit Visual"
[image10]:./examples/poly1.png "Fit Visual"
[image11]:./examples/poly2.png "Fit Visual"
[image6]: ./examples/lines.png "Output"
[video1]: ./project_video_out.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first and third code cells of the IPython notebook located in `./Advanced-Lane-Lines-3.ipynb` (or in lines 18 through 90 of the file called `Advanced-Lane-Lines-Copy2.py`).  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used different combination of color and gradient thresholds to generate a binary image. The code for this step is contained in the fourth and ninth code cells of the IPython notebook located in `./Advanced-Lane-Lines-3.ipynb` (or in lines 90 through 315 of the file called `Advanced-Lane-Lines-Copy2.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][image3]
![alt text][image7]

Final version of binarization: Sobel gradint on x coordinat, L and S channels of HLS layers of image representation, White/Yellow color filters and I determined the region of interest (without my car).

![alt text][image8]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `PerspTranform_()`, which appears in lines 345 through 400 in the file `Advanced-Lane-Lines-Copy2.py` (or, for example, in the 11th and 12th code cells of the IPython notebook).  The `PerspTranform_()` function takes as inputs an image (`img`).  I chose the source and destination points taking into account the image where the lines ase supposed to parallel and choose 4 dots on it.

This resulted in the following source and destination points:
     

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 595, 450      | 300, 0        | 
| 200, 720      | 300, 720      |
| 1100, 720     | 980, 720      |
| 685, 450      | 980, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]
![alt text][image9]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial. The code for this step is contained between 13th and 18th code cells of the IPython notebook located in `./Advanced-Lane-Lines-3.ipynb` (or in lines 400 through 600 of the file called `Advanced-Lane-Lines-Copy2.py`). The main function here is `find_window_centroids()` Here's an examples of my output for this step. 

![alt text][image5]
![alt text][image10]
![alt text][image11]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The code for this step is contained between 19th and 20th code cells of the IPython notebook located in `./Advanced-Lane-Lines-3.ipynb` (or in lines 600 through 630 of the file called `Advanced-Lane-Lines-Copy2.py`).

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

The code for this step is contained in 21 code cell of the IPython notebook located in `./Advanced-Lane-Lines-3.ipynb` (or in lines 630 through 660 of the file called `Advanced-Lane-Lines-Copy2.py`). Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_out.mp4)
---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

 Pipeline perform reasonably well, but there are several cases when it behave not as well as I expected:
 * Bright regions
 * Other cars driving near the lines
 I tried to improve this by color filters (yellow and white) but it is not enough. Also, the pipline works quite well on chelenge video but far away to be considered perfect in the hard chelenge.
