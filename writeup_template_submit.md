# Udacity Self Driving Car Nanodegree
## **Advanced Lane Finding Project**
---
The goals / steps of this project are the following:
* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

The videos with lane lines found are:

![video1](https://github.com/zoespot/CarND-Advanced-Lane-Lines/blob/master/output_videos/output_project_video.mp4)
![video2](https://github.com/zoespot/CarND-Advanced-Lane-Lines/blob/master/output_videos/output_challenge_video.mp4)

---
### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the 2nd code cell of the IPython notebook located in "detect-project.ipynb"   

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

![alt text](https://github.com/zoespot/CarND-Advanced-Lane-Lines/blob/master/writeup_images/camera_calibration.png)

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
[alt text]('https://github.com/zoespot/CarND-Advanced-Lane-Lines/blob/master/writeup_images/image_undist.png')

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (`binary_threshold` in 5th cell). I Here's an example of my output for this step. I tried extraction from HLS and HSV colorspaces and sobel gradients of X and Y directions. The most clear binary image is combination from S, L and V spaces.

[alt text]('https://github.com/zoespot/CarND-Advanced-Lane-Lines/blob/master/writeup_images/threshold_binary.png')

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The function `perspective_transform` is defined cell 7 in the IPython notebook).  The matching pairs of source points and destination points are listed in cell 16.  I chose the hardcode the source and destination points as below with try-and-error tunes

| Source        | Destination   |
|:-------------:|:-------------:|
| 313, 650     | 233, 650       |
| 1000, 650      | 1080, 650      |
| 720, 470    | 1080, 40    |
| 570, 470      | 233, 40        |

I verified that my perspective transform was working as expected by drawing the `pt[1-4]` and `dpt[1-4]` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image (use `straight_lines1` image).

[alt text]('https://github.com/zoespot/CarND-Advanced-Lane-Lines/blob/master/writeup_images/transform_points.png')

The actual meter per pixel are defined with respect to the pixels in the object space. They are used for real curvature and offset computation later.  
```
ym_per_pix = 30/Ypixel # meters per pixel in y dimension
xm_per_pix = 3.7/Xpixel# meters per pixel in x dimension
```

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I use sliding window for searching the lane lines from scratch (function `search_new_poly` in cell 9) and `search_around_poly` (cell 10) for searching lane lines with existing fitting from similar previous image. Both functions use 2nd polynomial to fit the nonzero points in the combine binary image after perspective transform.
In the pipeline, a function `find_lane_pixels` is used to choose when to reset and search again from scratch.
An example of fitted lane lines is as below:

[alt text]('https://github.com/zoespot/CarND-Advanced-Lane-Lines/blob/master/writeup_images/sliding_window.png')

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

It's part of `draw_area` function (in cell 13). It's computed with fitted lane lines with curvature formula in the course material. X,Y are actual meters converted from pixels. Final curvature is average of left and right lines. Offset is defined by distance between image center and middle point of left and right lane lines. The code is as below:
```
y_eval = np.max(ploty)*ym_per_pix
left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
left_curvature = (1+(2*left_fit_cr[0]*y_eval+left_fit_cr[1])**2)**1.5/(2*np.absolute(left_fit_cr[0]))
right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
right_curvature = (1+(2*right_fit_cr[0]*y_eval+right_fit_cr[1])**2)**1.5/(2*np.absolute(right_fit_cr[0]))

curvature = (left_curvature +right_curvature )/2/1000.0    
car_offset = (0.5*img.shape[1] - (np.average(left_fitx)+ np.average(right_fitx))/2) *xm_per_pix
```

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented in the function `draw_area`.  Here is an example of my result on a test image:

[alt text]('https://github.com/zoespot/CarND-Advanced-Lane-Lines/blob/master/writeup_images/draw_area.png')

#### 7. For challenge video, the ground are split with dark and light concrete. They will be detected with original algorithm as fake lane lines. Thresholds in new color space needs to be used to exclude them. LAB color space is added. B is good to detect yellow lines, and L is inverted to exclude the fake concrete lines in the sobel gradient extractions.

[alt text]('https://github.com/zoespot/CarND-Advanced-Lane-Lines/blob/master/writeup_images/lab.jpg')

[alt text]('https://github.com/zoespot/CarND-Advanced-Lane-Lines/blob/master/writeup_images/threshold_binary_challenge.png')

[alt text]('https://github.com/zoespot/CarND-Advanced-Lane-Lines/blob/master/writeup_images/challenge_draw_area.png')


---

### Pipeline (video)

#### 1. Provide a link to your final video output.  It performed reasonably well on the entire project video.

[video1](https://github.com/zoespot/CarND-Advanced-Lane-Lines/blob/master/output_videos/output_project_video.mp4)

[video2](https://github.com/zoespot/CarND-Advanced-Lane-Lines/blob/master/output_videos/output_challenge_video.mp4)

 #### 2. Pipeline with Lines Class
Line class is used to store the historical lane information. The advantages include:
* Skipped the bad frame with abrupt curvature change or left/right lane lines are not in parallel (usually from failure to detect right lines correctly)
* 10 recent lines are used to smooth the fit polynomial
* When bad frame appears, the algorithm resets to `search_new_poly` to detect with sliding windows from scratch; otherwise, it uses `search_around_poly` to search around existing lines and save time and resources.

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The pipeline fails most frequently with bad frames:
* I started debugging with extracting the individual images and analyzing frame by frame
  * I mainly tuned color space or gradient thresholds (or introduce new color space extraction) to improve the lane line detect in single image.
  * I also tuned the source and object points in the perspective transform to exclude the nonzero points outside the lane line region. The pixel to meter conversion parameters needs to be adjusted accordingly
* Then I utilized the `Line` Class properties to store or discard the bad frames
  * Discard frames either excessive polynomial fit changes or non-parallel curvatures or unable to detect one side lane Lines
  * Use the average of multiple frames to smooth the detected lane lines.
  * Use `Line` object to determine whether search lines from existing polynomial or restarting from scratch

#### 2. Challenge videos

In the challenge video case, adding LAB color space is most effective. I also add exceptions in the lane search for empty lanes. Further tuning the source and object points can also help improve the lane accuracy.

I also tried the harder challenge video, the v plane in the HSV color space works better for the bright background. However the existing sliding window algorithm works poorly in this video, since the lane lines curved to the left or right edges. I didn't find effective ways to handle this problem. Once the algorithm is found for highly curved lane lines, the video processing should be similar to the previous two videos.

#### 3. Final thoughts

* The optimal binary thresholds of specific color space and gradients change with circumstances. In real environment, we need to dynamically tune them with bright/dark lights, ground texture and weather conditions.
* It might be helpful to provide the processed lane line detected images to the convolutional neural network (CNN) as image augmentation as we did behavior cloning. It helps the CNN feature extraction layer more easily and correlate to the steering angle. With the lane lines on the images, the CNN may needs less parameters and can be trained and applied much faster and more accurately.
