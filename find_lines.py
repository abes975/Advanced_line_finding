import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import os
import pickle

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
    abs_sobel = absolute_sobel_thresh(undistorted, 'x', thresh=(30,120))
    #gray_images.append(('Sobel X', abs_sobel))
    mag_threshold = magnitude_thresh(undistorted, thresh=(30, 120))
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
    abs_sobel_R = absolute_sobel_thresh(R, 'x', thresh=(20,150))
    #B_list.append(('Sobel B', abs_sobel_B))
    #G_list.append(('Sobel G', abs_sobel_G))
    #R_list.append(('Sobel R', abs_sobel_R))

    #mag_threshold_B = magnitude_thresh(B, thresh=(20, 150))
    #mag_threshold_G = magnitude_thresh(G, thresh=(20, 150))
    mag_threshold_R = magnitude_thresh(R, thresh=(20, 150))
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
    abs_sobel_S = absolute_sobel_thresh(S, 'x', thresh=(20,150))
    #H_list.append(('Sobel H', abs_sobel_H))
    #L_list.append(('Sobel L', abs_sobel_L))
    #S_list.append(('Sobel S', abs_sobel_S))

    #mag_threshold_H = magnitude_thresh(H, thresh=(20, 150))
    #mag_threshold_L = magnitude_thresh(L, thresh=(20, 150))
    mag_threshold_S = magnitude_thresh(S, thresh=(20, 150))
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
    abs_sobel_V = absolute_sobel_thresh(V, 'x', thresh=(20,150))
    #H_list.append(('Sobel H', abs_sobel_H))
    #S_list.append(('Sobel S', abs_sobel_S))
    #V_list.append(('Sobel V', abs_sobel_V))

    #mag_threshold_H = magnitude_thresh(H, thresh=(20, 150))
    #mag_threshold_S = magnitude_thresh(S, thresh=(20, 150))
    mag_threshold_V = magnitude_thresh(V, thresh=(20, 150))
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
    #combined_S[ (abs_sobel_S == 1) | ((mag_threshold_S == 1) & (dir_threshold_S == 1))] = 1
    #H_list.append(('Combined H', combined_H))
    #S_list.append(('Combined S', combined_S))
    #V_list.append(('Combined V', combined_V))
    #thresholded_img.append(H_list)
    #thresholded_img.append(S_list)
    #thresholded_img.append(V_list)

    #display_result(non_thresholeded, thresholded_img)
    return combined_V




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
    # axes[0, 1].imshow(stacked_R_S_V)
    # axes[0, 1].set_title("Red, Saturation, Value")
    # axes[1, 0].imshow(stacked_g_S_V)
    # axes[1, 0].set_title("Gray, Saturation, Value")
    # axes[1, 1].imshow(stacked_g_V_R)
    # axes[1, 1].set_title("Gray, Value, Red")
    # plt.show()
    combined = np.zeros_like(combined_g)
    combined[(combined_g == 1) | (combined_R == 1) & ((combined_S == 1) | (combined_V == 1))] = 1
    plt.imshow(combined, cmap='gray')
    plt.show()

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
    test_images_path = "./test_images/*"
    test_images_files = glob.glob(test_images_path)
    for image_file in test_images_files:
        image = cv2.imread(image_file)
        process_image(image, mtx, dist)
