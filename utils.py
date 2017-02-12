import numpy as np
import cv2
import matplotlib.pyplot as plt

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
    Define src and dst points for perspecrive transformation
'''
def define_source_points():
    y_top = 460
    y_bot = 720
    x_left_bot = 203
    x_right_bot = 1127
    x_left_top = 585
    x_right_top = 695
# leftupperpoint = [585,460]
# rightupperpoint = [695,460]
# leftlowerpoint = [203,720]
# rightlowerpoint = [1127,720]
    points = np.float32([[x_left_top, y_top], [x_right_top, y_top], [x_right_bot, y_bot], [x_left_bot, y_bot]])
    return points

def define_dst_points():
    x_left_bot = 320
    x_left_top = 320
    x_right_bot = 960
    x_right_top = 900
    y_top = 0
    y_bot = 720
    return np.float32([[x_left_top, y_top],[x_right_top, y_top],[x_right_bot, y_bot],[x_left_bot, y_bot]])


'''
    Define a region of interest inside the image
    We will use the point of the region as the starting point for the
    perspective transform
'''
def extract_region_of_interest(img, points):
    mask_points = np.int32(points)

    #defining a blank mask to start with
    mask = np.zeros_like(img)
    ignore_mask_color = 1
    cv2.fillPoly(mask, [mask_points], ignore_mask_color)
    #plt.imshow(mask, cmap='gray')
    #plt.show()
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    # plt.imshow(masked_image, cmap='gray')
    # plt.show()
    return masked_image



# def diag_screen():
#     # middle panel text example
#     # using cv2 for drawing text in diagnostic pipeline.
#     font = cv2.FONT_HERSHEY_COMPLEX
#     middlepanel = np.zeros((120, 1280, 3), dtype=np.uint8)
#     cv2.putText(middlepanel, 'Estimated lane curvature: ERROR!', (30, 60), font, 1, (255,0,0), 2)
#     cv2.putText(middlepanel, 'Estimated Meters right of center: ERROR!', (30, 90), font, 1, (255,0,0), 2)
#
#
#     # assemble the screen example
#     diagScreen = np.zeros((1080, 1920, 3), dtype=np.uint8)
#     diagScreen[0:720, 0:1280] = mainDiagScreen
#     diagScreen[0:240, 1280:1600] = cv2.resize(diag1, (320,240), interpolation=cv2.INTER_AREA)
#     diagScreen[0:240, 1600:1920] = cv2.resize(diag2, (320,240), interpolation=cv2.INTER_AREA)
#     diagScreen[240:480, 1280:1600] = cv2.resize(diag3, (320,240), interpolation=cv2.INTER_AREA)
#     diagScreen[240:480, 1600:1920] = cv2.resize(diag4, (320,240), interpolation=cv2.INTER_AREA)*4
#     diagScreen[600:1080, 1280:1920] = cv2.resize(diag7, (640,480), interpolation=cv2.INTER_AREA)*4
#     diagScreen[720:840, 0:1280] = middlepanel
#     diagScreen[840:1080, 0:320] = cv2.resize(diag5, (320,240), interpolation=cv2.INTER_AREA)
#     diagScreen[840:1080, 320:640] = cv2.resize(diag6, (320,240), interpolation=cv2.INTER_AREA)
#     diagScreen[840:1080, 640:960] = cv2.resize(diag9, (320,240), interpolation=cv2.INTER_AREA)
#     diagScreen[840:1080, 960:1280] = cv2.resize(diag8, (320,240), interpolation=cv2.INTER_AREA)
