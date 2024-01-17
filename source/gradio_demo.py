"""Old version of the LAEO demo, new one is Head Pose Estimation and LAEO demo"""

import time
from pathlib import Path

import gdown
import gradio as gr

import logging
import os

import cv2
import numpy as np
import tensorflow as tf

from ai.detection import detect
from laeo_per_frame.interaction_per_frame_uncertainty import LAEO_computation
from utils.hpe import hpe, project_ypr_in2d
from utils.img_util import resize_preserving_ar, draw_detections, percentage_to_pixel, draw_key_points_pose, \
    visualize_vector

EXAMPLES= [['/home/federico/Music/1.jpg']]

def load_image(camera, ):
    # Capture the video frame by frame
    try:
        ret, frame = camera.read()
        return True, frame
    except:
        logging.Logger('Error reading frame')
        return False, None


def demo_play(img, laeo=True, rgb=False):
    # webcam in use

    # gpus = tf.config.list_physical_devices('GPU')

    # img = np.array(frame)
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

    # center_xy, yaw, pitch, roll = head_pose_estimation(kpt, 'centernet', gaze_model=gaze_model)

    # _________ extract hpe and print to img
    people_list = []

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
    if laeo:
        interaction_matrix = LAEO_computation(people_list, clipping_value=clip_uncertainty,
                                              clip=binarize_uncertainty)
    else:
        interaction_matrix = np.zeros((len(people_list), len(people_list)))
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
    return img


demo_webcam = gr.Interface(
        fn=demo_play,
        inputs=[gr.Image(source="webcam", streaming=True, every=0.8), # with no streaming-> acquire images
                gr.Checkbox(value=True, label="LAEO", info="Compute and display LAEO"),
                gr.Checkbox(value=False, label="rgb", info="Display output on W/B image"),
                ],
        outputs="image",
        live=True,
        title="Head Pose Estimation and LAEO",
        description="This is a demo developed by Federico Figari T. at MaLGa Lab, University of Genoa, Italy. You can choose to have only the Head Pose Estimation or also the LAEO computation (more than 1 person should be in the image). You need to take a picture and the algorithm will calculate the Head Pose and will be showed as an arrow on your face. LAEO, instead is showed colouring the arrow in green.",
        # examples=EXAMPLES,
        )

demo_upload = gr.Interface(
        fn=demo_play,
        inputs=[gr.Image(source="upload",), # with no streaming-> acquire images
                gr.Checkbox(value=True, label="LAEO", info="Compute and display LAEO"),
                gr.Checkbox(value=False, label="rgb", info="Display output on W/B image"),
                ],
        outputs="image",
        live=True,
        title="Head Pose Estimation and LAEO",
        description="This is a demo developed by Federico Figari T. at MaLGa Lab, University of Genoa, Italy. You can choose to have only the Head Pose Estimation or also the LAEO computation (more than 1 person should be in the image). You need to upload an image and the algorithm will calculate the Head Pose and will be showed as an arrow on your face. LAEO, instead is showed colouring the arrow in green.",
        examples=[[Path("/home/federico/Music/1.jpg")], [Path("/home/federico/Music/20.jpg")]]
        )

demo_tabbed = gr.TabbedInterface([demo_webcam, demo_upload], ["Demo from webcam", "Demo from upload"])

if __name__=='__main__':
    online = False
    if online:
        if not os.path.exists("data"):
            gdown.download_folder(
                    "https://drive.google.com/drive/folders/1nQ1Cb_tBEhWxy183t-mIcVH7AhAfa6NO?usp=drive_link",
                    use_cookies=False)
        gaze_model_path = 'data/head_pose_estimation'
        gaze_model = tf.keras.models.load_model(gaze_model_path, custom_objects={"tf": tf})
        path_to_model = 'data/keypoint_detector/centernet_hg104_512x512_kpts_coco17_tpu-32'
        model = tf.saved_model.load(os.path.join(path_to_model, 'saved_model'))
    else:
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

    demo_tabbed.queue(concurrency_count=1, max_size=10)
    demo_tabbed.launch()