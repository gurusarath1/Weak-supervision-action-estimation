'''
Step 2 -
Run this file to use the labeling functions to get labels for the unlabeled dataset.
'''

from snorkel.labeling import labeling_function
from snorkel.labeling import PandasLFApplier
from snorkel.labeling.model import LabelModel
import math
import numpy as np
import pandas as pd
from pose_recognition_settings import LEFT_HAND_ON_HEAD, LEFT_HAND_NOT_ON_HEAD


# Column name: Column index map
dataset_column_map = {'nose_x': 0, 'nose_y': 1, 'nose_z': 2, 'left_eye_inner_x': 3, 'left_eye_inner_y': 4, 'left_eye_inner_z': 5, 'left_eye_x': 6, 'left_eye_y': 7, 'left_eye_z': 8, 'left_eye_outer_x': 9, 'left_eye_outer_y': 10, 'left_eye_outer_z': 11, 'right_eye_inner_x': 12, 'right_eye_inner_y': 13, 'right_eye_inner_z': 14, 'right_eye_x': 15, 'right_eye_y': 16, 'right_eye_z': 17, 'right_eye_outer_x': 18, 'right_eye_outer_y': 19, 'right_eye_outer_z': 20, 'left_ear_x': 21, 'left_ear_y': 22, 'left_ear_z': 23, 'right_ear_x': 24, 'right_ear_y': 25, 'right_ear_z': 26, 'mouth_left_x': 27, 'mouth_left_y': 28, 'mouth_left_z': 29, 'mouth_right_x': 30, 'mouth_right_y': 31, 'mouth_right_z': 32, 'left_shoulder_x': 33, 'left_shoulder_y': 34, 'left_shoulder_z': 35, 'right_shoulder_x': 36, 'right_shoulder_y': 37, 'right_shoulder_z': 38, 'left_elbow_x': 39, 'left_elbow_y': 40, 'left_elbow_z': 41, 'right_elbow_x': 42, 'right_elbow_y': 43, 'right_elbow_z': 44, 'left_wrist_x': 45, 'left_wrist_y': 46, 'left_wrist_z': 47, 'right_wrist_x': 48, 'right_wrist_y': 49, 'right_wrist_z': 50, 'left_pinky_x': 51, 'left_pinky_y': 52, 'left_pinky_z': 53, 'right_pinky_x': 54, 'right_pinky_y': 55, 'right_pinky_z': 56, 'left_index_x': 57, 'left_index_y': 58, 'left_index_z': 59, 'right_index_x': 60, 'right_index_y': 61, 'right_index_z': 62, 'left_thumb_x': 63, 'left_thumb_y': 64, 'left_thumb_z': 65, 'right_thumb_x': 66, 'right_thumb_y': 67, 'right_thumb_z': 68, 'left_hip_x': 69, 'left_hip_y': 70, 'left_hip_z': 71, 'right_hip_x': 72, 'right_hip_y': 73, 'right_hip_z': 74, 'left_knee_x': 75, 'left_knee_y': 76, 'left_knee_z': 77, 'right_knee_x': 78, 'right_knee_y': 79, 'right_knee_z': 80, 'left_ankle_x': 81, 'left_ankle_y': 82, 'left_ankle_z': 83, 'right_ankle_x': 84, 'right_ankle_y': 85, 'right_ankle_z': 86, 'left_heel_x': 87, 'left_heel_y': 88, 'left_heel_z': 89, 'right_heel_x': 90, 'right_heel_y': 91, 'right_heel_z': 92, 'left_foot_index_x': 93, 'left_foot_index_y': 94, 'left_foot_index_z': 95, 'right_foot_index_x': 96, 'right_foot_index_y': 97, 'right_foot_index_z': 98}

def euclidean_distance(a, b):
    dist = np.linalg.norm(a - b)
    return dist

def unit_vector(vector):
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

#
#  LABELING FUNCTIONS - Heuristics to detect a particular pose
#
@labeling_function()
def left_hand_on_head_lf_angle_check_1(x):


    point11 = np.array((x[dataset_column_map['left_shoulder_x']], x[dataset_column_map['left_shoulder_y']], x[dataset_column_map['left_shoulder_z']]))
    point13 = np.array((x[dataset_column_map['left_elbow_x']], x[dataset_column_map['left_elbow_y']], x[dataset_column_map['left_elbow_z']]))
    point15 = np.array((x[dataset_column_map['left_wrist_x']], x[dataset_column_map['left_wrist_y']], x[dataset_column_map['left_wrist_z']]))

    vec1 = point11 - point13
    vec2 = point15 - point13

    angle = math.degrees(angle_between(vec1, vec2))
    if angle < 70:
        return LEFT_HAND_ON_HEAD
    else:
        return LEFT_HAND_NOT_ON_HEAD

@labeling_function()
def left_hand_on_head_lf_angle_check_2(x):

    point11 = np.array((x[dataset_column_map['left_shoulder_x']], x[dataset_column_map['left_shoulder_y']], x[dataset_column_map['left_shoulder_z']]))
    point13 = np.array((x[dataset_column_map['left_elbow_x']], x[dataset_column_map['left_elbow_y']], x[dataset_column_map['left_elbow_z']]))
    point23 = np.array((x[dataset_column_map['left_hip_x']], x[dataset_column_map['left_hip_x']], x[dataset_column_map['left_hip_z']]))

    vec1 = point11 - point13
    vec3 = point23 - point11

    angle = math.degrees(angle_between(-vec1, vec3))
    if angle > 90:
        return LEFT_HAND_ON_HEAD
    else:
        return LEFT_HAND_NOT_ON_HEAD

@labeling_function()
def dist1_heuristic(x):
    point7 = np.array((x[dataset_column_map['left_ear_x']], x[dataset_column_map['left_ear_y']], x[dataset_column_map['left_ear_z']]))
    point21 = np.array((x[dataset_column_map['left_thumb_x']], x[dataset_column_map['left_thumb_y']], x[dataset_column_map['left_thumb_z']]))

    if euclidean_distance(point7, point21) < 0.25:
        return LEFT_HAND_ON_HEAD
    else:
        return LEFT_HAND_NOT_ON_HEAD

@labeling_function()
def dist2_heuristic(x):
    point3 = np.array((x[dataset_column_map['left_eye_outer_x']], x[dataset_column_map['left_eye_outer_y']], x[dataset_column_map['left_eye_outer_z']]))
    point19 = np.array((x[dataset_column_map['left_index_x']], x[dataset_column_map['left_index_y']], x[dataset_column_map['left_index_z']]))

    if euclidean_distance(point3, point19) < 0.25:
        return LEFT_HAND_ON_HEAD
    else:
        return LEFT_HAND_NOT_ON_HEAD


if __name__ == '__main__':

    print('Running Weak Supervision using snorkel ...')

    # Labeling functions
    lfs = [left_hand_on_head_lf_angle_check_1, left_hand_on_head_lf_angle_check_2, dist1_heuristic, dist2_heuristic]
    # Apply each labeling function on the dataset get prediction of each labeling function
    applier = PandasLFApplier(lfs=lfs)
    train_set_unlabelled = pd.read_csv('unlabelled_set_left_hand_on_head.csv')
    L_train = applier.apply(df=train_set_unlabelled)
    print('labeling function prediction = ', L_train)

    # Weak supervision model to get a single prediction
    label_model = LabelModel(cardinality=2, verbose=True)
    label_model.fit(L_train=L_train, n_epochs=500, log_freq=100, seed=123)
    preds = label_model.predict(L_train)
    print('snorkel prediction = ', preds)

    # Write the output predictions to a file
    preds_string = ''
    for pred in preds:
        preds_string += str(pred) + '\n'
    with open('snorkel_preds.csv', 'w') as f:
        f.write(preds_string)






