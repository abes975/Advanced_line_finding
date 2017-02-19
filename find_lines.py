import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import os
import pickle
import math
import utils
import warper
from moviepy.editor import *

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
        mtx, dist = utils.calibrate_camera(calibration_filenames, grid_size=(9,6), visual=False)
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

    src_points = utils.define_source_points()
    dst_points = utils.define_dst_points()
    warper = warper.Warper(mtx, dist, src_points, dst_points, 15)

    # test_images_path = "./test_images/straight_lines2.jpg"
    # test_images_files = glob.glob(test_images_path)
    # for image_file in test_images_files:
    #     image = cv2.imread(image_file)
    #     out = warper.process_image(image)
    #     plt.imshow(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
    #     plt.show()

    # # Test videos
    # video_output = 'project_video_out.mp4'
    # # print('Starting video 1')
    # # clip1 = VideoFileClip("project_video.mp4")
    # # video_clip = clip1.fl_image(warper.process_image)
    # # video_clip.write_videofile(video_output, audio=False)
    video_output = 'project_video_out.mp4'
    if not os.path.exists(video_output):
        print('Starting video 1')
        clip1 = VideoFileClip("project_video.mp4")
        video_clip = clip1.fl_image(warper.process_image)
        video_clip.write_videofile(video_output, audio=False)
    # video_output = 'challenge_video_out.mp4'
    # if not os.path.exists(video_output):
    #     print('Starting video 2')
    #     clip1 = VideoFileClip("challenge_video.mp4")
    #     video_clip = clip1.fl_image(warper.process_image)
    #     video_clip.write_videofile(video_output, audio=False)
    # video_output = 'harder_challenge_video_out.mp4'
    # if not os.path.exists(video_output):
    #     print('Starting video 3')
    #     video_output = 'harder_challenge_video_out.mp4'
    #     clip1 = VideoFileClip("harder_challenge_video.mp4")
    #     video_clip = clip1.fl_image(warper.process_image)
    #     video_clip.write_videofile(video_output, audio=False)
