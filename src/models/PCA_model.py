import numpy as np
import joblib
from sklearn.decomposition import PCA

# Initialize a list to hold data_DC dictionaries for each fold
data_train_list = []  # Create an empty list to store training data_DC for each fold.

# Iterate through each fold (1 to 5)
for kf in range(1, 6):  # Loop through 5 folds, from 1 to 5.
    # Load the data_DC for the current fold
    data_read = joblib.load(f'../mid_data/{kf}-DATA.npy')  # Load data_DC from a .npy file for the current fold.

    # Perform PCA for dimensionality reduction
    pca_m = PCA(n_components=0.9980)  # Initialize PCA for 'm_data' to retain 99.80% variance.
    pca_d = PCA(n_components=0.9980)  # Initialize PCA for 'd_data' to retain 99.80% variance.

    # Apply PCA on 'm_data'
    data_d_after_pca = pca_m.fit_transform(data_read['d_data'])  # Fit and transform 'd_data' using PCA.

    # Apply PCA on 'd_data'
    data_t_after_pca = pca_d.fit_transform(data_read['t_data'])  # Fit and transform 't_data' using PCA.

    # Concatenate the PCA-transformed 'm_data' and 'd_data' based on the provided indices
    data_concat = np.concatenate((data_d_after_pca[data_read['index'][0]],  # Concatenate the transformed 'd_data'.
                                  data_t_after_pca[data_read['index'][1]]), axis=1)  # Concatenate the transformed 't_data'.

    # Split concatenated data_DC into training and testing sets
    train_data_concat = data_concat[data_read['train_idx']]  # Select training data_DC using training indices.
    test_data_concat = data_concat[data_read['test_idx']]  # Select testing data_DC using testing indices.

    # Create a dictionary to store training and testing data_DC along with labels
    data_dict = {
        'train_data': train_data_concat,  # Store the training data_DC.
        'test_data': test_data_concat,  # Store the testing data_DC.
        'y_train': data_read['y'][data_read['train_idx']],  # Store training labels.
        'y_test': data_read['y'][data_read['test_idx']],  # Store testing labels.
        'all_data': {
            'Em': data_read['d_data'].cpu().numpy(),  # Store original 'd_data'.
            'Ed': data_read['t_data'].cpu().numpy(),  # Store original 't_data'.
        },
    }
    # Append the data_DC dictionary to the list
    data_train_list.append(data_dict)  # Add the current fold's data_DC dictionary to the list.

data_dict = data_train_list[0]  # Select the data_DC dictionary for the first fold.

y_train = data_dict['y_train']  # Extract training labels from the first fold's data_DC dictionary.

# Print information about the training labels
print(f"y_train type: {type(y_train)}")  # Print the type of y_train.
print(f"y_train content: {y_train}")  # Print the content of y_train.

# Print information from the loaded data_DC
print(f"y_train type from loaded data: {type(data_read['y'])}")  # Print the type of labels from loaded data_DC.
print(f"y_train content from loaded data: {data_read['y']}")  # Print the content of labels from loaded data_DC.
