class Line():
    '''
        we will use these list to keep track of the previous values and
        use in case we don't find a good value from the picture
    '''
    def __init__(self, windows=10):
        self.left_fit = []
        self.right_fit = []
        self.lost_frames = 0
        self.windows = windows
        self.lost_frame_th = 5
        self.left_rad = -1
        self.right_rad = -1

    '''
        The following code is taken directly from the udacity lesson
        warped_img : one channel image resulting from the warpPerspective call
    '''
    def _find_lines_by_sliding_window(self, warped_img):
        # Assuming you have created a warped binary image called "binary_warped"
        # Take a histogram of the bottom half of the image
        # WHY ONLY HALF OF THE IMAGE?
        histogram = np.sum(warped_img[warped_img.shape[0]//2:,:], axis=0)
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
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50
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
            #print("Windows left has vertex (", win_xleft_left, ", ", win_y_high, ") and (", win_xleft_right, ", ", win_y_low, ")")
            # these are the left and the right vertex of the rectanle over x axe
            # For the window centerd at rightx_current
            win_xright_left = rightx_current - margin
            win_xright_right = rightx_current + margin
            #print("Windows right has vertex (", win_xright_left, ", ", win_y_high, ") and (", win_xright_right, ", ", win_y_low, ")")
            # Draw the windows on the visualization image
            #cv2.rectangle(out_img, (win_xleft_left, win_y_high),(win_xleft_right, win_y_low), (0,255,0), 2)
            #cv2.rectangle(out_img, (win_xright_left, win_y_high),(win_xright_right,win_y_low), (0,255,0), 2)
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_left) & (nonzerox < win_xleft_right)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_left) & (nonzerox < win_xright_right)).nonzero()[0]
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
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        self.left_fit.append(left_fit)
        self.right_fit.append(right_fit)
        ## Display image...but we have some problem now with libraries
        # # Generate x and y values for plotting
        # ploty = np.linspace(0, warped_img.shape[0]-1, warped_img.shape[0])
        # left_fitx = left_fit[0]* ploty**2 + left_fit[1] * ploty + left_fit[2]
        # right_fitx = right_fit[0]* ploty**2 + right_fit[1] * ploty + right_fit[2]
        #
        # out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        # out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        # plt.plot(left_fitx, ploty, color='yellow')
        # plt.plot(right_fitx, ploty, color='yellow')
        # plt.xlim(0, 1280)
        # plt.ylim(720, 0)
        # plt.imshow(out_img)
        # plt.show()

    def _find_lines_by_previous_values(self, warped_img, left_fit, rirght_fit):
        # Assume you now have a new warped binary image
        # from the next frame of video (also called "img")
        # It's now much easier to find line pixels!
        nonzero = warped_img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 100
        left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin)))
        right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        # Generate x and y values for plotting
        #ploty = np.linspace(0, warped_img.shape[0]-1, warped_img.shape[0])
        #left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        #right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        #And you're done! But let's visualize the result here as well
        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((warped_img, warped_img, warped_img))*255
        window_img = np.zeros_like(out_img)
        # Color in left and right line pixels
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
        plt.imshow(result)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)

    '''
        display information on the frame
            radius
            car's center offset
    '''
    def display_info(self, img):
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = "Radius of curvature is {:.2f}m".format(self.curvature)
        cv2.putText(img, text, (50, 50), font, 1, (255, 255, 255), 2)
        left_or_right = 'left' if self.car_offset < 0 else 'right'
        text = "Car is {:.2f}m {} off center".format(np.abs(self.car_offset), left_or_right)
        cv2.putText(img, text, (50, 100), font, 1, (255, 255, 255), 2)


    def calculate_radius(self):
        # Define y-value where we want radius of curvature
        ploty = np.linspace(0, 719, num=720)
        y_eval = np.max(ploty)
        # left_curverad = ((1 + (2 * self.left_fit[0] * y_eval +
        #         self.left_fit[1])**2)**1.5) / np.absolute(2 * self.left_fit[0])
        # right_curverad = ((1 + (2*self.right_fit[0]*y_eval +
        #         self.right_fit[1])**2)**1.5) / np.absolute(2*self.right_fit[0])
        # # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meters per pixel in x dimension

        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(ploty * ym_per_pix, leftx * xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty * ym_per_pix, rightx * xm_per_pix, 2)
        # Calculate the new radii of curvature
        left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix +
            left_fit_cr[1])**2)**1.5) / np.absolute(2 * left_fit_cr[0])
        right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix +
            right_fit_cr[1])**2)**1.5) / np.absolute(2 * right_fit_cr[0])
        # Now our radius of curvature is in meters
        self.left_rad = left_curverad,
        self.right_rad = right_curverad

    '''
        This will try to get the lines from the previous frame.
        For every frame we do thresholding and warping outside the class
    '''
    def process_next_frame(self, img):
        if self.lost_frame < self.lost_frame_th:
            # Just average and continue
            self._find_lines_by_previous_values(img)
        else:
            self._find_lines_by_sliding_window(img)
