from snorkel.labeling import labeling_function
from snorkel.labeling import PandasLFApplier
from snorkel.labeling.model import LabelModel
import math
import numpy as np
import pandas as pd

LEFT_HAND_ON_HEAD = 1
LEFT_HAND_NOT_ON_HEAD = 0

def unit_vector(vector):
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
@labeling_function()
def left_hand_on_head_lf_angle_check_1(x):

    vec1 = np.array((x[0], x[1], x[2]))
    vec2 = np.array((x[3], x[4], x[5]))

    angle = math.degrees(angle_between(vec1, vec2))
    if angle < 70:
        return LEFT_HAND_ON_HEAD
    else:
        return LEFT_HAND_NOT_ON_HEAD

@labeling_function()
def left_hand_on_head_lf_angle_check_2(x):

    vec1 = np.array((x[0], x[1], x[2]))
    vec3 = np.array((x[6], x[7], x[8]))

    angle = math.degrees(angle_between(-vec1, vec3))
    if angle > 90:
        return LEFT_HAND_ON_HEAD
    else:
        return LEFT_HAND_NOT_ON_HEAD

if __name__ == '__main__':

    print('Running Weak Supervision using snorkle ...')

    lfs = [left_hand_on_head_lf_angle_check_1, left_hand_on_head_lf_angle_check_2]

    applier = PandasLFApplier(lfs=lfs)

    train_set_unlabelled = pd.read_csv('unlabelled_set_left_hand_on_head.csv')

    L_train = applier.apply(df=train_set_unlabelled)

    print('Out = ', L_train)

    label_model = LabelModel(cardinality=2, verbose=True)
    label_model.fit(L_train=L_train, n_epochs=500, log_freq=100, seed=123)






