########################################################################
#
# Copyright (c) 2021, STEREOLABS.
#
# All rights reserved.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
########################################################################

"""
   This sample shows how to detect a human bodies and draw their
   modelised skeleton in an OpenGL window
"""
import argparse

import cv2
import pyzed.sl as sl
import source.ogl_viewer.viewer as gl
import source.cv_viewer.tracking_viewer as cv_viewer


def initialize_zed_camera(input_file=None):
    """Create a Camera object, set the configurations parameters and open the camera

    Returns:
        :param input_file:
        :return:
        :zed (pyzed.sl.Camera): Camera object
    """
    print(input_file)
    # Create a Camera object
    zed = sl.Camera()

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.depth_mode = sl.DEPTH_MODE.QUALITY  # depth mode (PERFORMANCE/QUALITY/ULTRA)
    init_params.coordinate_units = sl.UNIT.METER  # depth measurements (METER/CENTIMETER/MILLIMETER/FOOT/INCH)
    init_params.camera_resolution = sl.RESOLUTION.HD720  # resolution (HD720/HD1080/HD2K)
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP

    init_params.depth_minimum_distance = 0.40  # cm
    init_params.depth_maximum_distance = 15
    # init_params.depth_stabilization = False  # to improve computational performance

    # If applicable, use the SVO given as parameter
    # Otherwise use ZED live stream
    if input_file is not None:
        filepath = input_file
        print("Using SVO file: {0}".format(filepath))
        init_params.svo_real_time_mode = True
        init_params.set_from_svo_file(filepath)

    # Open the camera
    err = zed.open(init_params)
    if err!=sl.ERROR_CODE.SUCCESS:
        exit(1)

    # Create and set RuntimeParameters after opening the camera
    runtime_parameters = sl.RuntimeParameters()
    runtime_parameters.sensing_mode = sl.SENSING_MODE.FILL  # sensing mode (STANDARD/FILL)
    # Setting the depth confidence parameters
    runtime_parameters.confidence_threshold = 100
    runtime_parameters.textureness_confidence_threshold = 100

    # mirror_ref = sl.Transform()
    # mirror_ref.set_translation(sl.Translation(2.75, 4.0, 0))
    # tr_np = mirror_ref.m

    return zed, runtime_parameters

def load_image():
    pass


def infer_ypr():
    pass

def extract_keypoints_zedcam(zed):
    """Zed Cam starts and extracts keypoints with stereolabs SDK object detector.

    :param zed: (pyzed.sl.Camera): Camera object
    """
    # Enable Positional tracking (mandatory for object detection)
    positional_tracking_parameters = sl.PositionalTrackingParameters()
    # If the camera is static, uncomment the following line to have better performances and boxes sticked to the ground.
    positional_tracking_parameters.set_as_static = True

    zed.enable_positional_tracking(positional_tracking_parameters)

    obj_param = sl.ObjectDetectionParameters()
    obj_param.enable_body_fitting = True  # Smooth skeleton move
    obj_param.enable_tracking = True  # Track people across images flow
    obj_param.detection_model = sl.DETECTION_MODEL.HUMAN_BODY_FAST  # HUMAN_BODY_MEDIUM ->less fast but more accurate

    # Enable Object Detection module
    zed.enable_object_detection(obj_param)

    obj_runtime_param = sl.ObjectDetectionRuntimeParameters()
    obj_runtime_param.detection_confidence_threshold = 40

    # Get ZED camera information
    camera_info = zed.get_camera_information()

    # 2D viewer utilities
    display_resolution = sl.Resolution(min(camera_info.camera_resolution.width, 1280),
                                       min(camera_info.camera_resolution.height, 720))
    image_scale = [display_resolution.width / camera_info.camera_resolution.width
        , display_resolution.height / camera_info.camera_resolution.height]

    # Create OpenGL viewer
    viewer = gl.GLViewer()
    viewer.init(camera_info.calibration_parameters.left_cam, obj_param.enable_tracking)

    # Create ZED objects filled in the main loop
    bodies = sl.Objects()
    image = sl.Mat()

    while viewer.is_available():
        # Grab an image
        if zed.grab()==sl.ERROR_CODE.SUCCESS:
            # Retrieve left image
            zed.retrieve_image(image, sl.VIEW.LEFT, sl.MEM.CPU, display_resolution)
            # Retrieve objects
            zed.retrieve_objects(bodies, obj_runtime_param)


            for person in bodies.object_list:
                keypoints = person.keypoint_2d
                confidence = person.confidence
                bbox_3d = person.head_bounding_box

            # Update GL view
            viewer.update_view(image, bodies) # it draws stuff
            # Update OCV view
            image_left_ocv = image.get_data()
            cv_viewer.render_2D(image_left_ocv, image_scale, bodies.object_list, obj_param.enable_tracking)
            cv2.imshow("ZED | 2D View", image_left_ocv)
            cv2.waitKey(10)


    viewer.exit()
    image.free(sl.MEM.CPU)

    # Disable modules and close camera
    zed.disable_object_detection()
    zed.disable_positional_tracking()
    zed.close()

    with open('report_file.txt', 'w') as f:
        f.write(str(keypoints))
        f.write('/n')
        f.write(str(confidence))
        f.write('/n')
        f.write(str(bbox_3d))



def compute_laeo():
    pass


def save_files():
    pass

def myfunct():
    pass


# https://www.stereolabs.com/docs/api/python/classpyzed_1_1sl_1_1BODY__PARTS.html
# NOSE
# NECK
# RIGHT_SHOULDER
# RIGHT_ELBOW
# RIGHT_WRIST
# LEFT_SHOULDER
# LEFT_ELBOW
# LEFT_WRIST
# RIGHT_HIP
# RIGHT_KNEE
# RIGHT_ANKLE
# LEFT_HIP
# LEFT_KNEE
# LEFT_ANKLE
# RIGHT_EYE
# LEFT_EYE
# RIGHT_EAR
# LEFT_EAR




if __name__=="__main__":
    print("Running Body Tracking sample ... Press 'q' to quit")

    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", type=str, default=None, help="path to the model", required=True)
    ap.add_argument("-f", "--input-file", type=str, default=None, help="input a SVO file", required=False)
    config = ap.parse_args()

    print(config.input_file)
    zed, run_parameters = initialize_zed_camera(input_file=config.input_file)

    if str(config.model).lower() == 'zed':
        extract_keypoints_zedcam(zed=zed) # everything performed with stereilabs SDK
    elif str(config.model).lower() == 'centernet':
        print('centernet')
        raise NotImplementedError
    elif str(config.model).lower()=='openpose':
        print('openpose')
        raise NotImplementedError
    else:
        print('wrong input for model value')
        raise IOError # probably not correct error

