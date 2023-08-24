import math
import numpy as np
from ml_utils import angle_between, append_to_file, euclidean_distance


full_dataset_columns = ['nose_x','nose_y','nose_z','left_eye_inner_x','left_eye_inner_y','left_eye_inner_z','left_eye_x','left_eye_y','left_eye_z','left_eye_outer_x','left_eye_outer_y','left_eye_outer_z','right_eye_inner_x','right_eye_inner_y','right_eye_inner_z','right_eye_x','right_eye_y','right_eye_z','right_eye_outer_x','right_eye_outer_y','right_eye_outer_z','left_ear_x','left_ear_y','left_ear_z','right_ear_x','right_ear_y','right_ear_z','mouth_left_x','mouth_left_y','mouth_left_z','mouth_right_x','mouth_right_y','mouth_right_z','left_shoulder_x','left_shoulder_y','left_shoulder_z','right_shoulder_x','right_shoulder_y','right_shoulder_z','left_elbow_x','left_elbow_y','left_elbow_z','right_elbow_x','right_elbow_y','right_elbow_z','left_wrist_x','left_wrist_y','left_wrist_z','right_wrist_x','right_wrist_y','right_wrist_z','left_pinky_x','left_pinky_y','left_pinky_z','right_pinky_x','right_pinky_y','right_pinky_z','left_index_x','left_index_y','left_index_z','right_index_x','right_index_y','right_index_z','left_thumb_x','left_thumb_y','left_thumb_z','right_thumb_x','right_thumb_y','right_thumb_z','left_hip_x','left_hip_y','left_hip_z','right_hip_x','right_hip_y','right_hip_z','left_knee_x','left_knee_y','left_knee_z','right_knee_x','right_knee_y','right_knee_z','left_ankle_x','left_ankle_y','left_ankle_z','right_ankle_x','right_ankle_y','right_ankle_z','left_heel_x','left_heel_y','left_heel_z','right_heel_x','right_heel_y','right_heel_z','left_foot_index_x','left_foot_index_y','left_foot_index_z','right_foot_index_x','right_foot_index_y','right_foot_index_z']

landmark_list = \
['nose',
'left_eye_inner',
'left_eye',
'left_eye_outer',
'right_eye_inner',
'right_eye',
'right_eye_outer',
'left_ear',
'right_ear',
'mouth_left',
'mouth_right',
'left_shoulder',
'right_shoulder',
'left_elbow',
'right_elbow',
'left_wrist',
'right_wrist',
'left_pinky',
'right_pinky',
'left_index',
'right_index',
'left_thumb',
'right_thumb',
'left_hip',
'right_hip',
'left_knee',
'right_knee',
'left_ankle',
'right_ankle',
'left_heel',
'right_heel',
'left_foot_index',
'right_foot_index']

landmark_map = \
    {0: 'nose',
1:'left_eye_inner',
2:'left_eye',
3:'left_eye_outer',
4:'right_eye_inner',
5:'right_eye',
6:'right_eye_outer',
7:'left_ear',
8:'right_ear',
9:'mouth_left',
10:'mouth_right',
11:'left_shoulder',
12:'right_shoulder',
13:'left_elbow',
14:'right_elbow',
15:'left_wrist',
16:'right_wrist',
17:'left_pinky',
18:'right_pinky',
19:'left_index',
20:'right_index',
21:'left_thumb',
22:'right_thumb',
23:'left_hip',
24:'right_hip',
25:'left_knee',
26:'right_knee',
27:'left_ankle',
28:'right_ankle',
29:'left_heel',
30:'right_heel',
31:'left_foot_index',
32:'right_foot_index'}

# Function to visualize landmark skeleton on a 3D space
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

# Create a csv string with all the landmark co-ordinates
def create_datapoint_string(results):

    header = ''
    datapoint_string = ''
    header_dict = dict()

    dict_idx = 0
    for i, landmark in enumerate(results.pose_world_landmarks.landmark):
        datapoint_string += str(landmark.x) + ','
        datapoint_string += str(landmark.y) + ','
        datapoint_string += str(landmark.z) + ','

        header += '\'' + landmark_list[i] + '_x\','
        header += '\'' + landmark_list[i] + '_y\','
        header += '\'' + landmark_list[i] + '_z\','

        header_dict[landmark_list[i] + '_x'] = dict_idx
        dict_idx += 1
        header_dict[landmark_list[i] + '_y'] = dict_idx
        dict_idx += 1
        header_dict[landmark_list[i] + '_z'] = dict_idx
        dict_idx += 1

    datapoint_string += '\n'
    header += '\n'

    return datapoint_string, header, header_dict



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

    point7 = np.array((results.pose_world_landmarks.landmark[7].x, results.pose_world_landmarks.landmark[7].y,
              results.pose_world_landmarks.landmark[7].z))
    point3 = np.array((results.pose_world_landmarks.landmark[3].x, results.pose_world_landmarks.landmark[3].y,
              results.pose_world_landmarks.landmark[3].z))

    point21 = np.array((results.pose_world_landmarks.landmark[21].x, results.pose_world_landmarks.landmark[21].y,
              results.pose_world_landmarks.landmark[21].z))
    point19 = np.array((results.pose_world_landmarks.landmark[19].x, results.pose_world_landmarks.landmark[19].y,
              results.pose_world_landmarks.landmark[19].z))


    print(point11, point13, point15)

    vec1 = point11 - point13
    vec2 = point15 - point13
    vec3 = point23 - point11

    angle1 = math.degrees(angle_between(vec1, vec2))
    angle2 = math.degrees(angle_between(-vec1, vec3))
    print(f'angle1 = {angle1}')
    print(f'angle2 = {angle2}')
    print(f'dist1 = {euclidean_distance(point7, point21)}')
    print(f'dist2 = {euclidean_distance(point3, point19)}')

    # Create a csv string with all the landmark co-ordinates
    data_point_string = create_datapoint_string(results)
    # Add the data point to csv file
    append_to_file('unlabelled_set_left_hand_on_head.csv', data_point_string[0])