import time
from sklearn import svm, ensemble, neighbors
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import ParameterGrid
import numpy as np
from PCA_model import data_train_list

class ModelSelector:
    def __init__(self):
        # Initialize models and parameter grids
        self.models = {
            'svm': svm.SVC(probability=True),  # Support Vector Machine
            'rf': ensemble.RandomForestClassifier(),  # Random Forest
            'xgboost': XGBClassifier(),  # XGBoost
            'knn': neighbors.KNeighborsClassifier(),  # k-Nearest Neighbors
            'decision_tree': DecisionTreeClassifier(),  # Decision Tree
            'logistic_regression': LogisticRegression(),  # Logistic Regression
            'naive_bayes': GaussianNB()  # Naive Bayes
        }
        self.param_grids = {
            'svm': {'C': [100.0], 'kernel': ['rbf'], 'gamma': ['auto']},
            'rf': {'n_estimators': [200, 300, 400, 500], 'max_depth': [None, 10, 15, 20]},
            'xgboost': {'max_depth': [15], 'learning_rate': [0.1]},
            'knn': {'n_neighbors': [300]},
            'decision_tree': {'max_depth': [10]},
            'logistic_regression': {'C': [10.0]},
            'naive_bayes': [{}],
            'rf': {'n_estimators': [300], 'max_depth': [20]},
        }

    def get_models(self, model_list=[]):
        """
        Select models based on input list. If the list is empty, select all models.

        Parameters:
        model_list (list): List of model names to select.

        Returns:
        dict: Dictionary of selected models.
        """
        if not model_list:
            print('Selecting all models...')
            return self.models
        else:
            print('Selecting specific models...')
            return {key: self.models[key] for key in model_list}

    def train_with_grid_search(self, models_dict={}):
        """
        Train models using grid search with cross-validation.

        Parameters:
        models_dict (dict): Dictionary of models to train.

        Returns:
        dict: Dictionary containing the parameters and scores for each model.
        """
        if not models_dict:
            models_dict = self.models
        results_dict = {}

        for model_name, model in models_dict.items():
            results_dict[model_name] = {'params_scores': []}
            param_grid = self.param_grids[model_name]
            grid = ParameterGrid(param_grid)
            best_score = -1
            best_params = None

            for params in grid:
                scores = []
                cross_val_scores = []

                for kf in range(5):  # 5-fold cross-validation
                    data = data_train_list[kf]
                    X_train, X_test, y_train, y_test = data['train_data'], data['test_data'], data['y_train'], data['y_test']
                    model.set_params(**params)
                    model.fit(X_train, np.reshape(y_train, (-1,)))
                    y_score = model.predict_proba(X_test)
                    metrics = get_metrics(y_test, y_score[:, 1])

                    cross_val_scores.append(metrics)
                    scores.append(metrics[0][2])  # AUC score

                np.save("cross_val_results.npy", cross_val_scores)
                mean_score = np.mean(scores)
                results_dict[model_name]['params_scores'].append({'params': params, 'mean_score': mean_score})

                if mean_score > best_score:
                    best_score = mean_score
                    best_params = params

            print(f"Best parameters for {model_name}: {best_params}, Best AUC score: {best_score}")

        return results_dict


def get_metrics(real_score, predict_score):
    """
    Calculate performance metrics for binary classification models.

    Parameters:
    real_score (np.array): Actual scores.
    predict_score (np.array): Predicted scores.

    Returns:
    list: Calculated metrics and their names.
    """
    real_score, predict_score = real_score.flatten(), predict_score.flatten()
    sorted_predict_score = np.array(sorted(list(set(predict_score.flatten()))))
    thresholds = sorted_predict_score[np.int32(len(sorted_predict_score) * np.arange(1, 1000) / 1000)]
    thresholds = np.mat(thresholds)
    thresholds_num = thresholds.shape[1]

    predict_score_matrix = np.tile(predict_score, (thresholds_num, 1))
    negative_index = np.where(predict_score_matrix < thresholds.T)
    positive_index = np.where(predict_score_matrix >= thresholds.T)
    predict_score_matrix[negative_index] = 0
    predict_score_matrix[positive_index] = 1
    TP = predict_score_matrix.dot(real_score.T)
    FP = predict_score_matrix.sum(axis=1) - TP
    FN = real_score.sum() - TP
    TN = len(real_score.T) - TP - FP - FN

    fpr = FP / (FP + TN)
    tpr = TP / (TP + FN)
    ROC_dot_matrix = np.mat(sorted(np.column_stack((fpr, tpr)).tolist())).T
    ROC_dot_matrix.T[0] = [0, 0]
    ROC_dot_matrix = np.c_[ROC_dot_matrix, [1, 1]]

    x_ROC = ROC_dot_matrix[0].T
    y_ROC = ROC_dot_matrix[1].T
    auc = 0.5 * (x_ROC[1:] - x_ROC[:-1]).T * (y_ROC[:-1] + y_ROC[1:])

    recall_list = tpr
    precision_list = TP / (TP + FP)
    PR_dot_matrix = np.mat(sorted(np.column_stack((recall_list, precision_list)).tolist())).T
    PR_dot_matrix.T[0] = [0, 1]
    PR_dot_matrix = np.c_[PR_dot_matrix, [1, 0]]

    x_PR = PR_dot_matrix[0].T
    y_PR = PR_dot_matrix[1].T
    aupr = 0.5 * (x_PR[1:] - x_PR[:-1]).T * (y_PR[:-1] + y_PR[1:])

    f1_score_list = 2 * TP / (len(real_score.T) + TP - TN)
    accuracy_list = (TP + TN) / len(real_score.T)
    specificity_list = TN / (TN + FP)

    max_index = np.argmax(f1_score_list)
    f1_score = f1_score_list[max_index]
    accuracy = accuracy_list[max_index]
    specificity = specificity_list[max_index]
    recall = recall_list[max_index]
    precision = precision_list[max_index]

    print(f' auc:{auc[0, 0]:.4f}, aupr:{aupr[0, 0]:.4f}, f1_score:{f1_score:.4f}, accuracy:{accuracy:.4f}, recall:{recall:.4f}, specificity:{specificity:.4f}, precision:{precision:.4f}')
    return [real_score, predict_score, auc[0, 0], aupr[0, 0], f1_score, accuracy, recall, specificity, precision], \
           ['y_true', 'y_score', 'auc', 'prc', 'f1_score', 'acc', 'recall', 'specificity', 'precision']


def mult_func(num_layers):
    """
    Function to train models with grid search and save the results.

    Parameters:
    num_layers (int): Number of layers (not used in this function but can be part of the process).
    """
    all_list = []

    selector = ModelSelector()
    models = selector.get_models([])  # Select all models
    results_dict = selector.train_with_grid_search(models)  # Train models
    all_list.append(results_dict)

    save_path = "results.npy"
    np.save(save_path, all_list)

    print(all_list)
    return results_dict


if __name__ == '__main__':
    range_list = range(1, 2)

    import multiprocessing

    mut = True
    if mut:
        multiprocessing.set_start_method('spawn')

        pool = multiprocessing.Pool(processes=min(len(range_list), multiprocessing.cpu_count() - 1))

        items_to_process = range_list

        pool.map(mult_func, items_to_process)

        pool.close()
        pool.join()
