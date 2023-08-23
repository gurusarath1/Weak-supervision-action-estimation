import math
import numpy as np
from ml_utils import angle_between, append_to_file

# Source: https://github.com/Kazuhito00/mediapipe-python-sample/blob/main/sample_pose.py
def plot_world_landmarks(plt, ax, landmarks, visibility_th=0.5):
    landmark_point = []

    for index, landmark in enumerate(landmarks.landmark):
        landmark_point.append(
            [landmark.visibility, (landmark.x, landmark.y, landmark.z)])

    face_index_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    right_arm_index_list = [11, 13, 15, 17, 19, 21]
    left_arm_index_list = [12, 14, 16, 18, 20, 22]
    right_body_side_index_list = [11, 23, 25, 27, 29, 31]
    left_body_side_index_list = [12, 24, 26, 28, 30, 32]
    shoulder_index_list = [11, 12]
    waist_index_list = [23, 24]

    face_x, face_y, face_z = [], [], []
    for index in face_index_list:
        point = landmark_point[index][1]
        face_x.append(point[0])
        face_y.append(point[2])
        face_z.append(point[1] * (-1))

    right_arm_x, right_arm_y, right_arm_z = [], [], []
    for index in right_arm_index_list:
        point = landmark_point[index][1]
        right_arm_x.append(point[0])
        right_arm_y.append(point[2])
        right_arm_z.append(point[1] * (-1))

    left_arm_x, left_arm_y, left_arm_z = [], [], []
    for index in left_arm_index_list:
        point = landmark_point[index][1]
        left_arm_x.append(point[0])
        left_arm_y.append(point[2])
        left_arm_z.append(point[1] * (-1))

    right_body_side_x, right_body_side_y, right_body_side_z = [], [], []
    for index in right_body_side_index_list:
        point = landmark_point[index][1]
        right_body_side_x.append(point[0])
        right_body_side_y.append(point[2])
        right_body_side_z.append(point[1] * (-1))

    left_body_side_x, left_body_side_y, left_body_side_z = [], [], []
    for index in left_body_side_index_list:
        point = landmark_point[index][1]
        left_body_side_x.append(point[0])
        left_body_side_y.append(point[2])
        left_body_side_z.append(point[1] * (-1))

    shoulder_x, shoulder_y, shoulder_z = [], [], []
    for index in shoulder_index_list:
        point = landmark_point[index][1]
        shoulder_x.append(point[0])
        shoulder_y.append(point[2])
        shoulder_z.append(point[1] * (-1))

    waist_x, waist_y, waist_z = [], [], []
    for index in waist_index_list:
        point = landmark_point[index][1]
        waist_x.append(point[0])
        waist_y.append(point[2])
        waist_z.append(point[1] * (-1))

    ax.cla()
    ax.set_xlim3d(-1, 1)
    ax.set_ylim3d(-1, 1)
    ax.set_zlim3d(-1, 1)

    # ax.scatter(face_x, face_y, face_z)
    ax.plot(right_arm_x, right_arm_y, right_arm_z)
    ax.plot(left_arm_x, left_arm_y, left_arm_z)
    ax.plot(right_body_side_x, right_body_side_y, right_body_side_z)
    ax.plot(left_body_side_x, left_body_side_y, left_body_side_z)
    ax.plot(shoulder_x, shoulder_y, shoulder_z)
    ax.plot(waist_x, waist_y, waist_z)

    plt.pause(.001)

    return

def create_unlabelled_data_set(results, dataset_type='left_hand_on_head'):

    switch_case = {'left_hand_on_head': create_left_hand_on_head_data_set }

    switch_case[dataset_type](results)

#results = pose.process(image)
def create_left_hand_on_head_data_set(results):
    point11 = np.array((results.pose_world_landmarks.landmark[11].x, results.pose_world_landmarks.landmark[11].y,
              results.pose_world_landmarks.landmark[11].z))
    point13 = np.array((results.pose_world_landmarks.landmark[13].x, results.pose_world_landmarks.landmark[13].y,
              results.pose_world_landmarks.landmark[13].z))
    point15 = np.array((results.pose_world_landmarks.landmark[15].x, results.pose_world_landmarks.landmark[15].y,
              results.pose_world_landmarks.landmark[15].z))
    point23 = np.array((results.pose_world_landmarks.landmark[23].x, results.pose_world_landmarks.landmark[23].y,
              results.pose_world_landmarks.landmark[23].z))


    print(point11, point13, point15)

    vec1 = point11 - point13
    vec2 = point15 - point13
    vec3 = point23 - point11

    angle1 = math.degrees(angle_between(vec1, vec2))
    angle2 = math.degrees(angle_between(-vec1, vec3))
    print(f'angle1 = {angle1}')
    print(f'angle2 = {angle2}')

    data_point_string = f'{vec1[0]},{vec1[1]},{vec1[2]},{vec2[0]},{vec2[1]},{vec2[2]},{vec3[0]},{vec3[1]},{vec3[2]}\n'

    append_to_file('unlabelled_set_left_hand_on_head.csv', data_point_string)