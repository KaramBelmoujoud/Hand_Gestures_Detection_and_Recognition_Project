import mediapipe as mp
import cv2
import numpy as np
import copy
import itertools
from collections import deque, Counter
import tensorflow as tf


mp_hands = mp.solutions.hands # hands model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

def mediapipe_detection(image, model):
    image = cv2.flip(image, 1)                     # Mirror display
    debug_image = copy.deepcopy(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results ,debug_image

actions = ['stop','goLeft', 'goRight', 'modeDiaPo','modeNormal']

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # キーポイント
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point

def pre_process_Keypoint_history(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0]

    temp_point_history = copy.deepcopy(point_history)

    base_x, base_y = 0, 0
    for index, point in enumerate(temp_point_history):
        if index == 0:
            base_x, base_y = point[0], point[1]

        temp_point_history[index][0] = (temp_point_history[index][0] -
                                        base_x) / image_width
        temp_point_history[index][1] = (temp_point_history[index][1] -
                                        base_y) / image_height

    temp_point_history = list(
        itertools.chain.from_iterable(temp_point_history))

    return temp_point_history


class Classifier(object):
    def __init__(
        self,
        model_path='Model/Gestures_classifier.tflite',
        score_th=0.8,
        invalid_value=0,
        num_threads=1,
    ):
        self.interpreter = tf.lite.Interpreter(model_path=model_path,
                                               num_threads=num_threads)

        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.score_th = score_th
        self.invalid_value = invalid_value

    def __call__(
        self,
        point_history,
    ):
        input_details_tensor_index = self.input_details[0]['index']
        self.interpreter.set_tensor(
            input_details_tensor_index,
            np.array([point_history], dtype=np.float32))
        self.interpreter.invoke()

        output_details_tensor_index = self.output_details[0]['index']

        result = self.interpreter.get_tensor(output_details_tensor_index)

        result_index = np.argmax(np.squeeze(result))

        if np.squeeze(result)[result_index] < self.score_th:
            result_index = self.invalid_value

        return result_index



keypoint_classifier = Classifier()

# Parameters Initialisation 
history_length = 16 # lenght of list that takes max indexes of predections 
Keypoints_history = deque(maxlen=history_length)
Argmax_list = deque(maxlen=history_length)
use_boundary_recttangle = True

# Camera preparation
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

# Set mediapipe model 
with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5,max_num_hands=1) as hands: 

    while cap.isOpened():

       # Process Key (ESC: end) 
       key = cv2.waitKey(10)
       if key == 27:  # ESC
          break

       # Camera capture #####################################################
       ret, frame = cap.read()
       if not ret:
          break

       # Make detections
       image, results, debug_image = mediapipe_detection(frame, hands)

       if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):
          
                # Landmark calculation
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                
                # Conversion to relative coordinates / normalized coordinates
                pre_processed_Keypoints_list = pre_process_Keypoint_history(
                    debug_image, Keypoints_history)
                      
                Keypoints_history.append(landmark_list[12])            
                hand_sign_id=0       
                hand_sign_len = len(pre_processed_Keypoints_list)          
                if hand_sign_len == (history_length * 2):
                    hand_sign_id = keypoint_classifier(pre_processed_Keypoints_list)

                Argmax_list.append(hand_sign_id)
                most_common_fg_id = Counter(
                    Argmax_list).most_common()
                
                # Drawing part
                print(actions[most_common_fg_id[0][0]])
     

       # Screen reflection 
       cv2.imshow('Hand Gesture Recognition', debug_image)


cap.release()

cv2.destroyAllWindows()

        