import numpy as np
import joblib
from sklearn.decomposition import PCA

# Initialize a list to hold data dictionaries for each fold
data_train_list = []

# Iterate through each fold (1 to 5)
for kf in range(1, 6):
    # Load the data for the current fold
    data_read = joblib.load(f'mid_data/{kf}-DATA.npy')

    # Perform PCA for dimensionality reduction
    pca_m = PCA(n_components=0.9980)
    pca_d = PCA(n_components=0.9980)

    # Apply PCA on 'm_data'
    data_d_after_pca = pca_m.fit_transform(data_read['d_data'])

    # Apply PCA on 'd_data'
    data_t_after_pca = pca_d.fit_transform(data_read['t_data'])

    # Concatenate the PCA-transformed 'm_data' and 'd_data' based on the provided indices
    data_concat = np.concatenate((data_d_after_pca[data_read['index'][0]],
                                  data_t_after_pca[data_read['index'][1]]), axis=1)

    # Split concatenated data into training and testing sets
    train_data_concat = data_concat[data_read['train_idx']]
    test_data_concat = data_concat[data_read['test_idx']]

    # Create a dictionary to store training and testing data along with labels
    data_dict = {
        'train_data': train_data_concat,
        'test_data': test_data_concat,
        'y_train': data_read['y'][data_read['train_idx']],
        'y_test': data_read['y'][data_read['test_idx']],
        'all_data': {
            'Em': data_read['d_data'].cpu().numpy(),
            'Ed': data_read['t_data'].cpu().numpy(),
        },
    }

    # Define the filename for saving the data dictionary
    file_name = f"data_dict_fold_{kf}.dict"

    # Save the data dictionary using joblib
    joblib.dump(data_dict, file_name)

    # Append the data dictionary to the list
    data_train_list.append(data_dict)

data_dict = data_train_list[0]

y_train = data_dict['y_train']

print(f"y_train type: {type(y_train)}")
print(f"y_train content: {y_train}")

print(f"y_train type from loaded data: {type(data_read['y'])}")
print(f"y_train content from loaded data: {data_read['y']}")
