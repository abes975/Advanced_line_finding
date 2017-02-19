import numpy as np
import cv2
import matplotlib.pyplot as plt

class Line:
    '''
        we will use these list to keep track of the previous values and
        use in case we don't find a good value from the picture
    '''
    def __init__(self, windows=10, avg_count=5):
        # Initialize this to say we have not detected anything yet
        self.detected = False
        # we will keep current coefficient here and also a fit for the last
        # iteration
        self.left_fit = None
        self.right_fit = None
        # Here's the smoothed lines used to paint the road
        self.avg_left_x = None
        self.avg_right_x = None

        # number of sliding windows
        self.windows = windows

        # If we fail to detect (failed validation) we will increment the number
        # of lost frames...is we reach the treshold we start again with sliding
        # window
        self.lost_frames = 0
        self.lost_frame_th = 5

        # We will keep some already fitted lines to aveage in case we cannot
        # detect a new line on a frame
        self.avg_count = avg_count
        self.curr_count = 0
        self.iter_count = 0
        # Our images have 720 pixels as height
        self.left_x = np.zeros((self.avg_count,720))
        self.right_x = np.zeros((self.avg_count, 720))

        # Left and right curvature radius
        self.left_rad = -1
        self.right_rad = -1

    '''
        Given y values and coefficient return the x values for that fit
    '''
    def _poly_fitx(self, y, coeff):
        fit_x = coeff[0] * y ** 2 + coeff[1] * y + coeff[2]
        return fit_x

    # '''
    #     Need to add some validation for found coefficients
    # '''
    # def _validate(self):
    #     # Ie lines are almos parallel...
    #     # distance between lines is almost the right one..and so on...
    #     # Get slope for left and right and check if they are similar (means
    #     # that the line are parallels)
    #     # slope_left = (50 - 0) / (self.last_left_x[50] - self.last_left_x[0])
    #     # slope_right = (50 - 0) / (self.last_right_x[50] - self.last_right_x[0])
    #     # diff_slopes = abs(slope_left - slope_right)
    #     # print("Difference between slopes: ", diff_slopes, "slope_left ", slope_left, " slope_right ", slope_right)
    #     # if diff_slopes > 0.1:
    #     #     print("Diff solpes is too high ", diff_slopes)
    #     #     return False
    #     # Check distance in meter...we have 3.7 for USA between left and right lane
    #     xm_per_pix = 3.7/700
    #     diff1 = self.last_right_x[0] - self.last_left_x[0]
    #     diff2 = self.last_right_x[len(self.last_right_x)//2] - self.last_left_x[len(self.last_right_x)//2]
    #     diff3 = self.last_right_x[-1] - self.last_left_x[-1]
    #     #print("Diff1 ", diff1 * xm_per_pix, "diff2 = ", diff2 * xm_per_pix, " diff3 = ", diff3 * xm_per_pix)
    #     thresh = 700 * 1.05
    #     if diff1 > thresh:
    #         print("Diff1 is not ok")
    #         return False
    #         print("Diff2 is not ok")
    #     if diff2 > thresh:
    #         return False
    #     if diff3 > thresh:
    #         print("Diff3 is not ok")
    #         return False
    #     return True

    '''
        The following code is taken directly from the udacity lesson
        warped_img : one channel image resulting from the warpPerspective call
    '''
    def _find_lines_by_sliding_window(self, warped_img):
        # Assuming you have created a warped binary image called "binary_warped"
        # Take a histogram of the bottom third of the image
        histogram = np.sum(warped_img[warped_img.shape[0]//2:,:], axis=0)
        #plt.plot(histogram)
        # Create an output image to draw on and visualize the result
        out_img = np.dstack((warped_img, warped_img, warped_img)) * 255
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]/2)
        # Get the max of the histogram from left and right side to start
        leftx_base = np.argmax(histogram[:midpoint])
        # Get the biggest index in the second half of the array (but add midpoint)
        # as the index for argmax started again at position 0!!!
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Choose the number of sliding windows
        nwindows = self.windows
        # Set height of windows
        window_height = np.int(warped_img.shape[0]/nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = warped_img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Set the width of the windows +/- margin
        margin = 75
        # Set minimum number of pixels found to recenter window
        minpix = 30
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            # Y low is the low (as number) end of the windows..from img[height] to 0
            win_y_low = warped_img.shape[0] - (window + 1) * window_height
            win_y_high = warped_img.shape[0] - window * window_height
            # these are the left and the right vertex of the rectanle over x axe
            # For the window centerd at leftx_current
            win_xleft_left = leftx_current - margin
            win_xleft_right = leftx_current + margin
            #print("Windows left has vertex (", win_xleft_left, ", ", win_y_high,
            #   ") and (", win_xleft_right, ", ", win_y_low, ")")
            # these are the left and the right vertex of the rectanle over x axe
            # For the window centerd at rightx_current
            win_xright_left = rightx_current - margin
            win_xright_right = rightx_current + margin
            #print("Windows right has vertex (", win_xright_left, ", ", win_y_high,
            #   ") and (", win_xright_right, ", ", win_y_low, ")")
            # Draw the windows on the visualization image
            cv2.rectangle(out_img, (win_xleft_left, win_y_high),(win_xleft_right, win_y_low), (0,255,0), 2)
            cv2.rectangle(out_img, (win_xright_left, win_y_high),(win_xright_right,win_y_low), (0,255,0), 2)
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high)
                & (nonzerox >= win_xleft_left)
                & (nonzerox < win_xleft_right)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high)
                & (nonzerox >= win_xright_left)
                & (nonzerox < win_xright_right)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit a second order polynomial to each
        self.left_fit = np.polyfit(lefty, leftx, 2)
        self.right_fit  = np.polyfit(righty, rightx, 2)
        #
        # cv2.imshow("finding lines", out_img)
        # cv2.waitKey(0)
        ploty = np.linspace(0, warped_img.shape[0]-1, warped_img.shape[0])
        last_left_x = self._poly_fitx(ploty, self.left_fit)
        last_right_x = self._poly_fitx(ploty, self.right_fit)

        # out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        # out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        # plt.plot(left_fitx, ploty, color='yellow')
        # plt.plot(right_fitx, ploty, color='yellow')
        # plt.xlim(0, 1280)
        # plt.ylim(720, 0)
        # plt.imshow(out_img)
        # plt.show()
        self.detected = True
        return last_left_x, last_right_x


    def _find_lines_by_previous_values(self, warped_img):
        # Assume you now have a new warped binary image
        # from the next frame of video (also called "img")
        # It's now much easier to find line pixels!
        nonzero = warped_img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 75
        left_x = self._poly_fitx(nonzeroy, self.left_fit)
        right_x = self._poly_fitx(nonzeroy, self.right_fit)
        #left_lane_inds = ((nonzerox > (self.left_fit[0]*(nonzeroy**2) + self.left_fit[1]*nonzeroy + self.left_fit[2] - margin)) & (nonzerox < (self.left_fit[0]*(nonzeroy**2) + self.left_fit[1]*nonzeroy + self.left_fit[2] + margin)))
        left_lane_inds = ((nonzerox > (left_x - margin)) & (nonzerox < (left_x + margin)))
        right_lane_inds = ((nonzerox > (right_x - margin)) & (nonzerox < (right_x + margin)))

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        # Fit a second order polynomial to each
        self.left_fit = np.polyfit(lefty, leftx, 2)
        self.right_fit = np.polyfit(righty, rightx, 2)

        # Display image...but we have some problem now with libraries
        # Generate x and y values for plotting
        ploty = np.linspace(0, warped_img.shape[0]-1, warped_img.shape[0])
        last_left_x = self._poly_fitx(ploty, self.left_fit)
        last_right_x = self._poly_fitx(ploty, self.right_fit)

        return last_left_x, last_right_x

    '''
        display information on the frame
            radius
            car's center offset
    '''
    def display_info(self, img):
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = "Left Line Radius is {:.2f}m".format(self.left_rad)
        cv2.putText(img, text, (30, 30), font, 1, (255, 255, 255), 2)
        text = "Right Line Radius is {:.2f}m".format(self.right_rad)
        cv2.putText(img, text, (30, 60), font, 1, (255, 255, 255), 2)
        left_or_right = 'left' if self.car_offset < 0 else 'right'
        text = "Car is {:.2f}m {} off center".format(np.abs(self.car_offset), left_or_right)
        cv2.putText(img, text, (30, 90), font, 1, (255, 255, 255), 2)


    def paint_road(self, img):
        dst = np.zeros_like(img).astype(np.uint8)
        ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
        left_fit = self.avg_left_x
        right_fit = self.avg_right_x

        # Recast the x and y points into usable format for cv2.fillPoly()
        # Not so clear...but use it so far
        left = np.array([np.transpose(np.vstack([left_fit, ploty]))], np.int32)
        right = np.array([np.flipud(np.transpose(np.vstack([right_fit, ploty])))], np.int32)
        line_points = np.hstack((left, right))
        # Draw the lane onto the warped blank image
        cv2.fillPoly(dst, [line_points], (0,255, 0))
        return dst

    '''
        The code here inside is taken from Udacity lesson
    '''
    def _calculate_radius(self, point):
        # Define y-value where we want radius of curvature
        y_eval = point
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meters per pixel in x dimension

        # # Fit new polynomials to x,y in world space
        ploty = np.linspace(0, y_eval, num=y_eval)
        left_fit_cr = np.polyfit(ploty * ym_per_pix, self.avg_left_x * xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty * ym_per_pix, self.avg_right_x * xm_per_pix, 2)
        # Calculate the new radii of curvature
        left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix +
                        left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0] * y_eval * ym_per_pix +
                        right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
        # Now our radius of curvature is in meters
        self.left_rad = left_curverad
        self.right_rad = right_curverad

    def _car_position(self, width):
        # meters from center
        xm_per_pix = 3.7/700 # meteres per pixel in x dimension
        # Change this
        screen_middel_pixel = width/2
        left_lane_pixel = self.avg_left_x[0]    # x position for left lane
        right_lane_pixel = self.avg_right_x[0]   # x position for right lane
        car_middle_pixel = int((right_lane_pixel + left_lane_pixel)/2)
        screen_off_center = screen_middel_pixel-car_middle_pixel
        self.car_offset = xm_per_pix * screen_off_center


    '''
        This will try to get the lines from the previous frame.
        For every frame we do thresholding and warping outside the class
    '''
    def process_next_frame(self, img):
        if self.detected:
            # Just average and continue
            left_x, right_x = self._find_lines_by_previous_values(img)
        else:
            left_x, right_x = self._find_lines_by_sliding_window(img)

        i = (self.curr_count % self.avg_count)
        self.left_x[i] = left_x
        self.right_x[i] = right_x

        if (self.curr_count < (self.avg_count - 1)):
            self.avg_left_x = np.sum(self.left_x, axis=0) / (self.curr_count + 1)
            self.avg_right_x = np.sum(self.right_x, axis=0) / (self.curr_count + 1)
        else:
            self.avg_left_x = np.average(self.left_x, axis=0)
            self.avg_right_x = np.average(self.right_x, axis=0)

        self.curr_count += 1
        # Reset detection method
        if i == 0:
            self.detected = False

        self._calculate_radius(img.shape[0])
        self._car_position(img.shape[1])
