import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pose_recognition_utils import full_dataset_columns
from pose_recognition_settings import points_to_consider_for_left_hand_on_head, DEVICE
from final_predication_model_nn import Final_pred_nn
from ml_utils import save_torch_model
def create_new_filtered_csv_for_training(file_name, preds_file_name, points_to_consider_for_left_hand_on_head):
    train_set_unlabelled = pd.read_csv(file_name)
    train_set_unlabelled.columns = full_dataset_columns

    filtered_columns = []
    for col in points_to_consider_for_left_hand_on_head:
        filtered_columns.append(col + '_x')
        filtered_columns.append(col + '_y')
        filtered_columns.append(col + '_z')

    filtered_train_set = train_set_unlabelled[filtered_columns]

    preds_df = pd.read_csv(preds_file_name)
    location = len(points_to_consider_for_left_hand_on_head) * 3
    filtered_train_set.insert(loc=location, column='preds', value=preds_df)

    return filtered_train_set.dropna()




if __name__ == '__main__':
    dataset = create_new_filtered_csv_for_training('unlabelled_set_left_hand_on_head.csv', 'snorkle_preds.csv', points_to_consider_for_left_hand_on_head)

    dataset_numpy = np.array(dataset)

    x = torch.Tensor(dataset_numpy[:,0:-1]).to(DEVICE)
    y = torch.Tensor(dataset_numpy[:,-1]).unsqueeze(1).to(DEVICE)

    print(x.shape)
    print(y.shape)

    inputs_len = len(points_to_consider_for_left_hand_on_head) * 3
    net = Final_pred_nn(inputs_len, 1, 50).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()

    # create your optimizer
    optimizer = optim.Adam(net.parameters(), lr=0.0001)

    net.train()
    for i in range(4000):
        # in your training loop:
        optimizer.zero_grad()  # zero the gradient buffers
        preds = net(x)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()  # Does the update

        acc = torch.sum(((preds > 0.5) == y)) / preds.shape[0]

        print(f'loss = {loss.item()}')
        print(f'Acc = {acc.item()}')

    save_torch_model(net, two_copies=False)


