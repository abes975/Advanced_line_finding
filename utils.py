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


# def car_position():
#     # meters from center
#     xm_per_pix = 3.7/700 # meteres per pixel in x dimension
#     screen_middel_pixel = img.shape[1]/2
#     left_lane_pixel = lane_info[6][0]    # x position for left lane
#     right_lane_pixel = lane_info[5][0]   # x position for right lane
#     car_middle_pixel = int((right_lane_pixel + left_lane_pixel)/2)
#     screen_off_center = screen_middel_pixel-car_middle_pixel
#     meters_off_center = xm_per_pix * pixels_off_center
