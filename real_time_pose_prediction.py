'''
Step 4 -
Run this file to do real-time prediction on videos using the trained model.
'''

import cv2
import mediapipe as mp
from pose_recognition_settings import DRAW_FLAG, points_to_consider_for_left_hand_on_head, DEVICE
from pose_recognition_utils import landmark_list
from ml_utils import *
from final_predication_model_nn import Final_pred_nn
import torch.nn as nn
from pose_recognition_settings import LEFT_HAND_ON_HEAD, LEFT_HAND_NOT_ON_HEAD

if __name__ == '__main__':

    print('Running Pose Prediction ...')

    inputs_len = len(points_to_consider_for_left_hand_on_head) * 3
    net = Final_pred_nn(inputs_len, 1, 50)
    # Load the trained model
    load_torch_model(net, load_latest=False)
    net.to(DEVICE)
    net.eval()

    plt_3d_axs = plt.axes(projection='3d')

    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    # VIDEO
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
        while cap.isOpened():
            ret, frame = cap.read()

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make detection
            results = pose.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.pose_landmarks:
                if DRAW_FLAG:
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                # Get the co-ordinates and run it through the pose prediction model
                data_point_for_prediction_neural_net = [] # x
                for landmark in points_to_consider_for_left_hand_on_head:
                    landmark_index = landmark_list.index(landmark)
                    data_point_for_prediction_neural_net.append(results.pose_world_landmarks.landmark[landmark_index].x)
                    data_point_for_prediction_neural_net.append(results.pose_world_landmarks.landmark[landmark_index].y)
                    data_point_for_prediction_neural_net.append(results.pose_world_landmarks.landmark[landmark_index].z)

                x = torch.Tensor(data_point_for_prediction_neural_net).to(DEVICE)
                pred = nn.Sigmoid()(net(x))

                if pred.item() == LEFT_HAND_ON_HEAD:
                    print('Left HAND ON HEAD !!')
                else:
                    print('Left HAND NOT ON HEAD !!')

            cv2.imshow('Real time prediction running ...', image)

            if cv2.waitKey(10) and 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
