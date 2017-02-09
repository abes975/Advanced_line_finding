import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import os
import pickle
import math

'''
    Will calibrate the camera using the image passed as input
    input is image in BRG colorspace and grid_size is a tuple with number of
    corners in the image.
    ### Is it possible to mix different sizes of grid while calibrating the
    ## same camera? not clear here...asked on the forum no one answered...
'''
def calibrate_camera(file_names, grid_size=(9,6), visual=False):
    objp = np.zeros((grid_size[0] * grid_size[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:grid_size[0], 0:grid_size[1]].T.reshape(-1,2)
    #print("objp[0,::],  ", objp)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.rare
    calibration_images = glob.glob(file_names)
    for cal_img in calibration_images:
        img = cv2.imread(cal_img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, grid_size, None)
        if ret == True:
            # useless but it's nice to display corners on the image just to check
            # visually if we are on the right track
            img = cv2.drawChessboardCorners(img, grid_size, corners, ret)
            if visual == True:
                cv2.imshow(cal_img, img)
                cv2.waitKey(0)
            # Get the matrix to undistort the image
            objpoints.append(objp)
            imgpoints.append(corners)
        else:
            print("Did not find any corner for image ", cal_img)
    if visual == True:
        cv2.destroyAllWindows()
    # Here we calculate the coefficient for our set...and then we will return them
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    return mtx, dist


'''
debug function used to display images
'''
def display_result(original, processed):
    # We have also the original image
    if original is None or processed is None or len(processed) == 0:
        return
    # Take the first element as example
    rows = len(processed[0]) + 1
    columns = len(original)
    print("We have ", rows, " rows and ", columns , "columnss")
    f, axes = plt.subplots(rows, columns)
    f.tight_layout()
    axes[0, 0].imshow(cv2.cvtColor(original[0][1], cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title(original[0][0])
    for j in range(1, columns):
        axes[0, j].imshow(original[j][1], cmap='gray')
        axes[0, j].set_title(original[j][0])
    # Hide unused pictures
    for i in range(1, rows):
        axes[i, 0].axis('off')
    for i in range(0, rows - 1):
        for j in range(columns-1):
            print(" i + 1 ", i+1, " j + 1", j+1)
            axes[i+1, j+1].imshow(processed[j][i][1], cmap='gray')
            axes[i+1, j+1].set_title(processed[j][i][0])
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()

'''
    apply sobel operator to the image
    param img source image
    param orientation which axis x, y calculate sobel absolute threshold
    param th_min minimum value to keep
    param th_max maximum value to keep
    returns thresholded image
'''
def absolute_sobel_thresh(image, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Calculate directional gradient
    if orient == 'x':
      sobel = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    else:
      sobel = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Apply threshold
    absolute = np.sqrt(sobel**2)
    norm = np.uint8(255 * absolute/np.max(absolute))
    grad_binary = np.zeros_like(norm)
    grad_binary[(norm > thresh[0]) & ( norm <= thresh[1]) ] = 1
    return grad_binary

'''
    calculate magnitude threshold
'''
def magnitude_thresh(image, sobel_kernel=3, thresh=(0, 255)):
    # Calculate gradient magnitude
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobel_xy = np.sqrt(sobel_x ** 2 + sobel_y **2)
    norm = np.uint8(abs_sobel_xy * 255 / np.max(abs_sobel_xy))
    # Apply threshold
    mag_binary = np.zeros_like(norm)
    mag_binary[(norm > thresh[0]) & (norm <= thresh[1]) ] = 1
    return mag_binary

'''
    calculate gradient direction
'''
def direction_thresh(image, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Calculate gradient direction
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobel_x = np.sqrt(sobel_x **2)
    abs_sobel_y = np.sqrt(sobel_y **2)
    dir_grad = np.arctan2(abs_sobel_y, abs_sobel_x)
    # Apply threshold
    dir_binary = np.zeros_like(dir_grad)
    dir_binary[ (dir_grad > thresh[0]) & (dir_grad <= thresh[1]) ] = 1
    return dir_binary



'''
This process gray image
'''
def process_gray(img, mtx, dist):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    undistorted = cv2.undistort(gray, mtx, dist, None, mtx)
    #gray_images = []
    #thresholded_img = []
    #non_thresholeded = [ ('Original', img), ('Gray',gray)]
    abs_sobel = absolute_sobel_thresh(undistorted, 'x', thresh=(30,150))
    #gray_images.append(('Sobel X', abs_sobel))
    mag_threshold = magnitude_thresh(undistorted, thresh=(30, 150))
    #gray_images.append(('Magnitude ', mag_threshold))
    dir_threshold = direction_thresh(undistorted, sobel_kernel=15, thresh=(0.7, 1.3))
    #gray_images.append(('Direction', dir_threshold))
    combined = np.zeros_like(gray)
    combined[(abs_sobel == 1) | (mag_threshold == 1) & (dir_threshold == 1)] = 1
    #gray_images.append(('Combined', combined))
    #thresholded_img.append(gray_images)
    #display_result(non_thresholeded, thresholded_img)
    return combined
'''
This process single channel in BGR image
'''
def process_brg(img, mtx, dist):
    undistorted = cv2.undistort(img, mtx, dist, None, mtx)
    #B = undistorted[:,:,0]
    #G = undistorted[:,:,1]
    R = undistorted[:,:,2]
    #thresholded_img = []
    #R_list = []
    #G_list = []
    #B_list = []
    #non_thresholeded = [ ('Original', img), ('Blu Chan',B),
    #                    ('Green Chan', G), ('Red Chan', R)]

    #abs_sobel_B = absolute_sobel_thresh(B, 'x', thresh=(20,150))
    #abs_sobel_G = absolute_sobel_thresh(G, 'x', thresh=(20,150))
    abs_sobel_R = absolute_sobel_thresh(R, 'x', thresh=(30,150))
    #B_list.append(('Sobel B', abs_sobel_B))
    #G_list.append(('Sobel G', abs_sobel_G))
    #R_list.append(('Sobel R', abs_sobel_R))

    #mag_threshold_B = magnitude_thresh(B, thresh=(20, 150))
    #mag_threshold_G = magnitude_thresh(G, thresh=(20, 150))
    mag_threshold_R = magnitude_thresh(R, thresh=(30, 150))
    #B_list.append(('Magnitude B', mag_threshold_B))
    #G_list.append(('Magnitude G', mag_threshold_G))
    #R_list.append(('Magnitude R', mag_threshold_R))

    #dir_threshold_B = direction_thresh(B, sobel_kernel=15, thresh=(0.7, 1.2))
    #dir_threshold_G = direction_thresh(G, sobel_kernel=15, thresh=(0.7, 1.2))
    dir_threshold_R = direction_thresh(R, sobel_kernel=15, thresh=(0.7, 1.2))
    #B_list.append(('Direction B', dir_threshold_B))
    #G_list.append(('Direction G', dir_threshold_G))
    #R_list.append(('Direction R', dir_threshold_R))

    combined_R = np.zeros_like(dir_threshold_R)
    combined_R[ (abs_sobel_R == 1) | ((mag_threshold_R == 1) & (dir_threshold_R == 1))] = 1
    #combined_G = np.zeros_like(dir_threshold_G)
    #combined_G[ (abs_sobel_G == 1) | ((mag_threshold_G == 1) & (dir_threshold_G == 1))] = 1
    #combined_B = np.zeros_like(dir_threshold_B)
    #combined_B[ (abs_sobel_B == 1) | ((mag_threshold_B == 1) & (dir_threshold_B == 1))] = 1
    #R_list.append(('Combined R', combined_R))
    #G_list.append(('Combined G', combined_G))
    #B_list.append(('Combined B', combined_B))

    #thresholded_img.append(B_list)
    #thresholded_img.append(G_list)
    #thresholded_img.append(R_list)
    #display_result(non_thresholeded, thresholded_img)
    return combined_R


def process_hls(img, mtx, dist):
    undistorted = cv2.undistort(img, mtx, dist, None, mtx)
    hls = cv2.cvtColor(undistorted, cv2.COLOR_RGB2HLS)
    #H = hls[:,:,0]
    #L = hls[:,:,1]
    S = hls[:,:,2]
    #thresholded_img = []
    #H_list = []
    #S_list = []
    #L_list = []
    #non_thresholeded = [ ('Original', img), ('H Chan',H),
    #                    ('L Chan', L), ('S Chan', S)]

    #abs_sobel_H = absolute_sobel_thresh(H, 'x', thresh=(20,150))
    #abs_sobel_L = absolute_sobel_thresh(L, 'x', thresh=(20,150))
    abs_sobel_S = absolute_sobel_thresh(S, 'x', thresh=(175,250))
    #H_list.append(('Sobel H', abs_sobel_H))
    #L_list.append(('Sobel L', abs_sobel_L))
    #S_list.append(('Sobel S', abs_sobel_S))

    #mag_threshold_H = magnitude_thresh(H, thresh=(20, 150))
    #mag_threshold_L = magnitude_thresh(L, thresh=(20, 150))
    mag_threshold_S = magnitude_thresh(S, thresh=(175, 250))
    #H_list.append(('Magnitude H', mag_threshold_H))
    #L_list.append(('Magnitude L', mag_threshold_H))
    #S_list.append(('Magnitude S', mag_threshold_S))

    #dir_threshold_H = direction_thresh(H, sobel_kernel=15, thresh=(0.7, 1.2))
    #dir_threshold_L = direction_thresh(L, sobel_kernel=15, thresh=(0.7, 1.2))
    dir_threshold_S = direction_thresh(S, sobel_kernel=15, thresh=(0.7, 1.2))
    #H_list.append(('Direction H', dir_threshold_H))
    #L_list.append(('Direction L', dir_threshold_H))
    S_list.append(('Direction S', dir_threshold_S))


    combined_S = np.zeros_like(dir_threshold_S)
    combined_S[ (abs_sobel_S == 1) | ((mag_threshold_S == 1) & (dir_threshold_S == 1))] = 1
    #combined_H = np.zeros_like(dir_threshold_H)
    #combined_H[ (abs_sobel_H == 1) | ((mag_threshold_H == 1) & (dir_threshold_H == 1))] = 1
    #combined_L = np.zeros_like(dir_threshold_L)
    #combined_L[ (abs_sobel_L == 1) | ((mag_threshold_L == 1) & (dir_threshold_L == 1))] = 1
    #H_list.append(('Combined H', combined_H))
    #L_list.append(('Combined L', combined_L))
    #S_list.append(('Combined S', combined_S))
    #thresholded_img.append(H_list)
    #thresholded_img.append(L_list)
    #thresholded_img.append(S_list)

    #display_result(non_thresholeded, thresholded_img)
    return combined_S

def process_hsv(img, mtx, dist):
    undistorted = cv2.undistort(img, mtx, dist, None, mtx)
    hsv = cv2.cvtColor(undistorted, cv2.COLOR_RGB2HSV)
    #H = hsv[:,:,0]
    #S = hsv[:,:,1]
    V = hsv[:,:,2]
    #thresholded_img = []
    #H_list = []
    #S_list = []
    V_list = []
    #non_thresholeded = [ ('Original', img), ('H Chan',H),
    #                    ('S Chan', S), ('V Chan', V)]

    #abs_sobel_H = absolute_sobel_thresh(H, 'x', thresh=(20,150))
    #abs_sobel_S = absolute_sobel_thresh(S, 'x', thresh=(20,150))
    abs_sobel_V = absolute_sobel_thresh(V, 'x', thresh=(175,250))
    #H_list.append(('Sobel H', abs_sobel_H))
    #S_list.append(('Sobel S', abs_sobel_S))
    #V_list.append(('Sobel V', abs_sobel_V))

    #mag_threshold_H = magnitude_thresh(H, thresh=(20, 150))
    #mag_threshold_S = magnitude_thresh(S, thresh=(20, 150))
    mag_threshold_V = magnitude_thresh(V, thresh=(175, 250))
    #H_list.append(('Magnitude H', mag_threshold_H))
    #S_list.append(('Magnitude S', mag_threshold_S))
    V_list.append(('Magnitude V', mag_threshold_V))

    #dir_threshold_H = direction_thresh(H, sobel_kernel=15, thresh=(0.7, 1.2))
    #dir_threshold_S = direction_thresh(S, sobel_kernel=15, thresh=(0.7, 1.2))
    dir_threshold_V = direction_thresh(V, sobel_kernel=15, thresh=(0.7, 1.2))
    #H_list.append(('Direction H', dir_threshold_H))
    #S_list.append(('Direction S', dir_threshold_S))
    #V_list.append(('Direction V', dir_threshold_V))


    combined_V = np.zeros_like(dir_threshold_V)
    combined_V[ (abs_sobel_V == 1) | ((mag_threshold_V == 1) & (dir_threshold_V == 1))] = 1
    #combined_H = np.zeros_like(dir_threshold_H)
    #combined_H[ (abs_sobel_H == 1) | ((mag_threshold_H == 1) & (dir_threshold_H == 1))] = 1
    #combined_S = np.zeros_like(dir_threshold_S)
    #combined_S[ (abs_sobetop
    #display_result(non_thresholeded, thresholded_img)
    return combined_V


'''
    Define a region of interest inside the image
    We will use the point of the region as the starting point for the
    perspective transform
'''
def extract_region_of_interest(img):
    height = img.shape[0]
    width = img.shape[1]
    # y_top = math.ceil(height/10 * 6.2)
    # y_bot = math.ceil(height/10 * 9)
    # x_left_bot = math.ceil(width/6 * 1)
    # x_right_bot = math.ceil(width/6 * 5)
    # x_left_top = math.ceil(width/10 * 4.6)
    # x_right_top = math.ceil(width/10 * 5)
    #print("X_l_t =", x_left_top, "X_l_b = ", x_left_bot, "X_r_t = ", x_right_top,
    #    "X_r_b = ", x_right_bot, "y_top = ", y_top, " y_bot = ", y_bot)
    # The order is important as we will use thos points to calculate destination
    # points...
    y_top = 460
    y_bot = 720
    x_left_bot = 203
    x_right_bot = 1180
    x_left_top = 585
    x_right_top = 715
    points = np.int32([[x_left_top, y_top], [x_right_top, y_top], [x_right_bot, y_bot], [x_left_bot, y_bot]])

    #defining a blank mask to start with
    mask = np.zeros_like(img)
    ignore_mask_color = 1
    cv2.fillPoly(mask, [points], ignore_mask_color)
    # plt.imshow(mask, cmap='gray')
    # plt.show()
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    # plt.imshow(masked_image, cmap='gray')
    # plt.show()
    return np.float32(points), masked_image


def find_destination_points(src_points):
    x_left_bot = 400
    x_left_top = 400
    x_right_bot = 980
    x_right_top = 980
    y_top = 0
    y_bot = 720
    return np.float32([[x_left_top, y_top],[x_right_top, y_top],[x_right_bot, y_bot],[x_left_bot, y_bot]])


def find_lines(binary_warped, win):
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[math.ceil(binary_warped.shape[0]/2):,:], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
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
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2)
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
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

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    plt.imshow(out_img)
    plt.show()


# # Assume you now have a new warped binary image
# # from the next frame of video (also called "img")
# # It's now much easier to find line pixels!
# nonzero = img.nonzero()
# nonzeroy = np.array(nonzero[0])
# nonzerox = np.array(nonzero[1])
# margin = 100
# left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin)))
# right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))
#
# # Again, extract left and right line pixel positions
# leftx = nonzerox[left_lane_inds]
# lefty = nonzeroy[left_lane_inds]
# rightx = nonzerox[right_lane_inds]
# righty = nonzeroy[right_lane_inds]
# # Fit a second order polynomial to each
# left_fit = np.polyfit(lefty, leftx, 2)
# right_fit = np.polyfit(righty, rightx, 2)
# # Generate x and y values for plotting
# ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
# left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
# right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
# And you're done! But let's visualize the result here as well
# # Create an image to draw on and an image to show the selection window
# out_img = np.dstack((img, img, img))*255
# window_img = np.zeros_like(out_img)
# # Color in left and right line pixels
# out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
# out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
#
# # Generate a polygon to illustrate the search window area
# # And recast the x and y points into usable format for cv2.fillPoly()
# left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
# left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
# left_line_pts = np.hstack((left_line_window1, left_line_window2))
# right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
# right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
# right_line_pts = np.hstack((right_line_window1, right_line_window2))
#
# # Draw the lane onto the warped blank image
# cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
# cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
# result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
# plt.imshow(result)
# plt.plot(left_fitx, ploty, color='yellow')
# plt.plot(right_fitx, ploty, color='yellow')
# plt.xlim(0, 1280)
# plt.ylim(720, 0)

'''
    This is the main routine.
    We will undistort the image and then treshold it
    and then...
'''
def process_image(img, mtx, dist):
    # Decide here what to do later...
    #sobel_gray, magnitude_gray, direction_gray = process_gray(img, mtx, dist)
    combined_g = process_gray(img, mtx, dist)
    combined_R = process_brg(img,mtx, dist)
    combined_S = process_hsv(img,mtx, dist)
    combined_V = process_hsv(img,mtx, dist)
    #stacked_g_R_S = np.dstack((combined_g, combined_R, combined_S))
    #stacked_R_S_V = np.dstack((combined_R, combined_S, combined_V))
    #stacked_g_S_V = np.dstack((combined_g, combined_S, combined_V))
    #stacked_g_V_R = np.dstack((combined_g, combined_V, combined_R))
    # f, axes = plt.subplots(2, 2)
    # f.tight_layout()
    # axes[0, 0].imshow(stacked_g_R_S)
    # axes[0, 0].set_title("Gray, Red, Saturation")
    # axes[0, 1].imshow(stacked_R_give in to meValue, Red")
    # plt.show()
    combined = np.zeros_like(combined_g)
    combined[(combined_g == 1) | (combined_R == 1) & ((combined_S == 1) | (combined_V == 1))] = 1
    src_points, masked_image = extract_region_of_interest(combined)
    dst_points = find_destination_points(src_points)
    # plt.imshow(combined, cmap='gray')
    # plt.show()

    # now do a prospective transform
    # we have to define destination points...y is aligned and I does not change for me
    # I just translate to the origin nothing more :)

    M = cv2.getPerspectiveTransform(src_points, dst_points)
    Minv = cv2.getPerspectiveTransform(dst_points, src_points)
    #print("Src points = ", src_points, " dst points ", dst_points, "M" , M)
    warped = cv2.warpPerspective(np.float64(masked_image), M, (combined.shape[1], combined.shape[0]))
    #plt.imshow(warped, cmap='gray')
    #plt.show()
    inv_warp = cv2.warpPerspective(np.float64(warped), Minv, (combined.shape[1], combined.shape[0]))
    # plt.imshow(inv_warp, cmap='gray')
    # plt.show()
    find_lines(warped, 10)



if __name__ == "__main__":
    matrices_file = "./camera_calibration_matrices.p"
    if os.path.exists(matrices_file):
        print("We find camera calibration parameters so we load them")
        with open(matrices_file, 'rb') as pickle_file:
            dist_pickle = pickle.load(pickle_file)
        mtx = dist_pickle['mtx']
        dist = dist_pickle['dist']
    else:
        print("Camera calibration in progress....")
        calibration_filenames = "./camera_cal/calibration*.jpg"
        mtx, dist = calibrate_camera(calibration_filenames, grid_size=(9,6), visual=False)
        # Save the result as pickle file
        dist_pickle = {}
        dist_pickle['mtx'] = mtx
        dist_pickle['dist'] = dist
        pickle.dump(dist_pickle, open( "./camera_calibration_matrices.p", "wb" ) )
        print("Saved camera calibration matrices")
    # Use at first the same images we use to calibrate as test images...
    # test_images = glob.glob(calibration_filenames)
    # for distorted in test_images:
    #     dist_img = cv2.imread(distorted)
    #     undistorted = cv2.undistort(dist_img, mtx, dist, None, mtx)
    #     dest_file = "/tmp/undistorted_" + os.path.basename(distorted)
    #     cv2.imwrite(dest_file, undistorted)
    #     cv2.imshow(distorted, undistorted)
    #     cv2.waitKey(0)
    # cv2.destroyAllWindows()

    #distorted = "./camera_cal/calibration4.jpg"
    #dist_img = cv2.imread(distorted)
    #undistorted = cv2.undistort(dist_img, mtx, dist, None, mtx)
    #dest_file = "./output_images/undistorted4.jpg"
    #cv2.imwrite(dest_file, undistorted)
    #cv2.imshow(distorted, undistorted)
    #cv2.waitKey(0)
    test_images_path = "./test_images/straight_lines2.jpg"
    test_images_files = glob.glob(test_images_path)
    for image_file in test_images_files:
        image = cv2.imread(image_file)
        process_image(image, mtx, dist)
