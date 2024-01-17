"""
   This sample shows how to detect a human bodies and draw their
   modelised skeleton in an window
   Calculate the direction of view form face keypoints and draw the view line.
   The line is coloured: GREEN -> high ocular interaction, BLACk -> low interaction
"""
import argparse
import logging
import os

import cv2
import numpy as np
import tensorflow as tf


from ai.detection import detect
from ai.tracker import Sort
from laeo_per_frame.interaction_per_frame_uncertainty import LAEO_computation
from utils.hpe import hpe, project_ypr_in2d
from utils.img_util import resize_preserving_ar, draw_detections, percentage_to_pixel, draw_key_points_pose, \
    visualize_vector, draw_key_points_pose_zedcam

# from utils.my_utils import retrieve_xyz_from_detection, compute_distance, save_key_points_to_json
# from utils.my_utils import normalize_wrt_maximum_distance_point, retrieve_interest_points



def load_image(camera, ):
    # Capture the video frame by frame
    try:
        ret, frame = camera.read()
        return True, frame
    except:
        logging.Logger('Error reading frame')
        return False, None


def compute_laeo():
    pass


def save_files():
    pass


def myfunct():
    pass

def demo_play():
    # webcam in use
    # define a video capture object
    camera = cv2.VideoCapture(0)

    mot_tracker = Sort(max_age=20, min_hits=1, iou_threshold=0.4)

    print('load hpe')
    gaze_model_path = '/home/federico/Documents/Models_trained/head_pose_estimation'
    gaze_model = tf.keras.models.load_model(gaze_model_path, custom_objects={"tf": tf})

    print('start centernet keypoint extractor')
    # path_to_model = '/media/DATA/Users/Federico/centernet_hg104_512x512_kpts_coco17_tpu-32'
    # path to your centernet model: https://tfhub.dev/tensorflow/centernet/resnet50v2_512x512/1
    path_to_model = '/home/federico/Documents/Models_trained/keypoint_detector/centernet_hg104_512x512_kpts_coco17_tpu-32'
    if not os.path.isdir(path_to_model):
        path_to_model = '/home/federico/Documents/Models_trained/keypoint_detector/centernet_hg104_512x512_kpts_coco17_tpu-32'
        if not os.path.isdir(path_to_model):
            raise IOError('path for model is incorrect, cannot find centernet model')

    model = tf.saved_model.load(os.path.join(path_to_model, 'saved_model'))

    input_shape_od_model = (512, 512)
    # params
    min_score_thresh, max_boxes_to_draw, min_distance = .45, 50, 1.5

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    # gpus = tf.config.list_physical_devices('GPU')

    flag, img = load_image(camera)

    while flag:
        # Use Flip code 0 to flip vertically, 1 to flip horizontally and -1 to flip both
        img = cv2.flip(img, 1)
        # do something

        # with tf.device(gpus[0]):

        # img = np.array(frame)
        rgb = False
        if not rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        img_resized, new_old_shape = resize_preserving_ar(img, input_shape_od_model)

        print('inference centernet')
        detections, elapsed_time = detect(model, img_resized, min_score_thresh,
                                          new_old_shape)  # detection classes boxes scores
        # probably to draw on resized
        img_with_detections = draw_detections(img_resized, detections, max_boxes_to_draw, None, None, None)
        # cv2.imshow("aa", img_with_detections)

        det, kpt = percentage_to_pixel(img.shape, detections['detection_boxes'], detections['detection_scores'],
                                       detections['detection_keypoints'], detections['detection_keypoint_scores'])

        # tracker stuff
        trackers = mot_tracker.update(det, kpt)
        people = mot_tracker.get_trackers()

        # center_xy, yaw, pitch, roll = head_pose_estimation(kpt, 'centernet', gaze_model=gaze_model)

        # _________ extract hpe and print to img
        people_list = []
        vector_list = []

        print('inferece hpe')

        for j, kpt_person in enumerate(kpt):
            yaw, pitch, roll, tdx, tdy = hpe(gaze_model, kpt_person, detector='centernet')

            # img = draw_axis_3d(yaw[0].numpy()[0], pitch[0].numpy()[0], roll[0].numpy()[0], image=img, tdx=tdx, tdy=tdy,
            #                    size=50)

            people_list.append({'yaw'      : yaw[0].numpy()[0],
                                'yaw_u'    : 0,
                                'pitch'    : pitch[0].numpy()[0],
                                'pitch_u'  : 0,
                                'roll'     : roll[0].numpy()[0],
                                'roll_u'   : 0,
                                'center_xy': [tdx, tdy]
                                })

        for i in range(len(det)):
            img = draw_key_points_pose(img, kpt[i])

        # call LAEO
        clip_uncertainty = 0.5
        binarize_uncertainty = False
        interaction_matrix = LAEO_computation(people_list, clipping_value=clip_uncertainty,
                                              clip=binarize_uncertainty)
        # coloured arrow print per person
        # TODO coloured arrow print per person
        for index, person in enumerate(people_list):
            green = round((max(interaction_matrix[index, :])) * 255)
            colour = (0, green, 0)
            if green < 40:
                colour = (0, 0, 255)
            vector = project_ypr_in2d(person['yaw'], person['pitch'], person['roll'])
            img = visualize_vector(img, person['center_xy'], vector, title="",
                                   color=colour)
        cv2.namedWindow('MaLGa Lab Demo')  # , cv2.WINDOW_NORMAL
        cv2.imshow('MaLGa Lab Demo', img)

        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if cv2.waitKey(1) & 0xFF==ord('q'):
            break
        # Update the flag and frame as last step of the loop
        flag, img = load_image(camera)

    # After the loop release the cap object
    camera.release()
    # Destroy all the windows
    cv2.destroyAllWindows()

def extract_keypoints_centernet(path_to_model, zed, gaze_model_path, rgb=False):
    """

    :param model:
    :param zed:
    """

    model = tf.saved_model.load(os.path.join(path_to_model, 'saved_model'))
    # model = tf.keras.models.load_model(gaze_model_path, custom_objects={"tf": tf})

    input_shape_od_model = (512, 512)

    image, image_cpu, depth_image, point_cloud = sl.Mat(), sl.Mat(), sl.Mat(), sl.Mat()

    # params
    min_score_thresh, max_boxes_to_draw, min_distance = .45, 50, 1.5

    # camera_info = zed.get_camera_information()
    # display_resolution = sl.Resolution(min(camera_info.camera_resolution.width, 1280),
    #                                    min(camera_info.camera_resolution.height, 720))
    # call HPE
    print('load hpe')
    gaze_model = tf.keras.models.load_model(gaze_model_path, custom_objects={"tf": tf})

    # tracker stuff
    mot_tracker = Sort(max_age=20, min_hits=1, iou_threshold=0.4)

    while zed.grab() == sl.ERROR_CODE.SUCCESS:
        # Retrieve left image
        print('image retrieved')
        # if tf.test.is_gpu_available():
        #     zed.retrieve_image(image, sl.VIEW.LEFT, sl.MEM.GPU, display_resolution)
        #     zed.retrieve_image(image_cpu, sl.VIEW.LEFT, sl.MEM.CPU, display_resolution)
        # else:
        zed.retrieve_image(image_cpu, sl.VIEW.LEFT, sl.MEM.CPU, display_resolution)
            # image_cpu = image.get_pointer()
        # Retrieve objects
        # zed.retrieve_objects(bodies, obj_runtime_param)

        img = np.array(image_cpu.get_data()[:, :, :3])
        if not rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        img_resized, new_old_shape = resize_preserving_ar(img, input_shape_od_model)

        print('inference centernet')
        detections, elapsed_time = detect(model, img_resized, min_score_thresh,
                                          new_old_shape)  # detection classes boxes scores
        # probably to draw on resized
        img_with_detections = draw_detections(img_resized, detections, max_boxes_to_draw, None, None, None)
        # cv2.imshow("aa", img_with_detections)

        det, kpt = percentage_to_pixel(img.shape, detections['detection_boxes'], detections['detection_scores'],
                                       detections['detection_keypoints'], detections['detection_keypoint_scores'])

        # tracker stuff
        trackers = mot_tracker.update(det, kpt)
        people = mot_tracker.get_trackers()

        # center_xy, yaw, pitch, roll = head_pose_estimation(kpt, 'centernet', gaze_model=gaze_model)

        # _________ extract hpe and print to img
        people_list = []
        vector_list = []

        print('inferece hpe')

        for j, kpt_person in enumerate(kpt):
            yaw, pitch, roll, tdx, tdy = hpe(gaze_model, kpt_person, detector='centernet')

            # img = draw_axis_3d(yaw[0].numpy()[0], pitch[0].numpy()[0], roll[0].numpy()[0], image=img, tdx=tdx, tdy=tdy,
            #                    size=50)

            people_list.append({'yaw': yaw[0].numpy()[0],
                                'yaw_u': 0,
                                'pitch': pitch[0].numpy()[0],
                                'pitch_u': 0,
                                'roll': roll[0].numpy()[0],
                                'roll_u': 0,
                                'center_xy': [tdx, tdy]})

        for i in range(len(det)):
            img = draw_key_points_pose(img, kpt[i])
        # img = draw_axis(yaw[i], pitch[i], roll[i], image=img, tdx=center_xy[0], tdy=center_xy[1], size=50) #single person

        # call LAEO
        clip_uncertainty = 0.5
        binarize_uncertainty = False
        interaction_matrix = LAEO_computation(people_list, clipping_value=clip_uncertainty, clip=binarize_uncertainty)
        # coloured arrow print per person


        print('before cv2')

        for index, person in enumerate(people_list):
            red = green = round((max(interaction_matrix[index, :])) * 255)
            colour = (0, green, 255-green)
            vector = project_ypr_in2d(person['yaw'], person['pitch'], person['roll'])
            img = visualize_vector(img, person['center_xy'], vector, title="", color=colour)
        cv2.namedWindow('MaLGa Lab Demo', cv2.WINDOW_NORMAL)
        cv2.imshow('MaLGa Lab Demo', img)
        # cv2.resizeWindow('MaLGa Lab Demo', 400, 400)

        print('after cv2')

        try:
            laeo_1, laeo_2 = (np.unravel_index(np.argmax(interaction_matrix, axis=None), interaction_matrix.shape))
            # print something around face
        except:
            pass
        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # save_key_points_to_json(kpts, path_json + ".json")

        # XYZ = retrieve_xyz_from_detection(detections['detection_boxes_centroid'], pc_img)
        # _, violate, couple_points = compute_distance(XYZ, min_distance)
        # img_with_violations = draw_detections(img, detections, max_boxes_to_draw, violate, couple_points)

    print('image free GPU/CPU')
    image_cpu.free(sl.MEM.CPU)
    # image.free(sl.MEM.GPU)
    cv2.destroyAllWindows()

    # Disable modules and close camera
    zed.close()


if __name__ == "__main__":
    """Example of usage:
            -m zed
            [-f /media/DATA/Users/Federico/Zed_Images/HD720_SN24782978_14-06-59.svo]
        or 
            -m centernet
            [-f /your_file]
        m: identifies the keypoints extractor algorithm
        f: a pre-recorded zedcam file, .svo format"""

    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", type=str, default=None, help="path to the model", required=True)
    ap.add_argument("-f", "--input-file", type=str, default=None, help="input a video file, implementation to yet completed", required=False)
    config = ap.parse_args()

    # choose between real time and pre-recorded file
    if config.input_file is not None:
        print('video file {}'.format(config.input_file))
        raise  NotImplementedError('video file input not yet implemented source')
    else:
        print('real time camera acquisition')

    demo_play()