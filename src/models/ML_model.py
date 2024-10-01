import time
from sklearn import svm, ensemble, neighbors
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import ParameterGrid
import numpy as np
from PCA_model import data_train_list
import multiprocessing

class ModelSelector:
    def __init__(self):
        # Initializing a dictionary of models and their parameter grids.
        self.models = {
            # 'svm': svm.SVC(probability=True),  # Support Vector Machine (commented out).
            'rf': ensemble.RandomForestClassifier(),  # Random Forest model.
            # Other models are commented out.
        }
        self.param_grids = {
            # 'svm': {'C': [100.0], 'kernel': ['rbf'], 'gamma': ['auto']},  # Parameters for SVM (commented out).
            'rf': {'n_estimators': [300], 'max_depth': [20]},  # Parameters for Random Forest.
        }

    def get_models(self, model_list=[]):
        """
        Select models based on input list. If the list is empty, select all models.
        """
        if not model_list:
            print('Selecting all models...')  # Inform user of model selection.
            return self.models  # Return all models.
        else:
            print('Selecting specific models...')  # Inform user of specific model selection.
            return {key: self.models[key] for key in model_list}  # Return selected models.

    def train_with_grid_search(self, models_dict={}):
        """
        Train models using grid search with cross-validation.
        """
        if not models_dict:
            models_dict = self.models  # Use all models if none specified.
        results_dict = {}  # Initialize results storage.

        for model_name, model in models_dict.items():
            results_dict[model_name] = {'params_scores': []}  # Store results for the model.
            param_grid = self.param_grids[model_name]  # Get parameters for the model.
            grid = ParameterGrid(param_grid)  # Create grid for parameter combinations.
            best_score = -1  # Initialize best score.
            best_params = None  # Initialize best parameters.

            for params in grid:
                scores = []  # List to store scores for this parameter combination.
                cross_val_scores = []  # List to store cross-validation scores.

                for kf in range(5):  # Loop for 5-fold cross-validation.
                    data = data_train_list[kf]  # Get training/testing data_DC for the fold.
                    X_train, X_test, y_train, y_test = data['train_data'], data['test_data'], data['y_train'], data['y_test']
                    model.set_params(**params)  # Set model parameters.
                    model.fit(X_train, np.reshape(y_train, (-1,)))  # Fit model to training data_DC.
                    y_score = model.predict_proba(X_test)  # Get predicted probabilities.
                    metrics = get_metrics(y_test, y_score[:, 1])  # Calculate performance metrics.

                    cross_val_scores.append(metrics)  # Append metrics for this fold.
                    scores.append(metrics[0][2])  # Append AUC score.

                np.save("cross_val_results.npy", cross_val_scores)  # Save cross-validation results.
                mean_score = np.mean(scores)  # Calculate mean AUC score for this parameter set.
                results_dict[model_name]['params_scores'].append({'params': params, 'mean_score': mean_score})  # Store score and params.

                if mean_score > best_score:  # Check if this score is the best.
                    best_score = mean_score  # Update best score.
                    best_params = params  # Update best parameters.

            print(f"Best parameters for {model_name}: {best_params}, Best AUC score: {best_score}")  # Output results for the model.

        return results_dict  # Return all results.

def get_metrics(real_score, predict_score):
    """
    Calculate performance metrics for binary classification models.
    """
    real_score, predict_score = real_score.flatten(), predict_score.flatten()  # Flatten input arrays.
    sorted_predict_score = np.array(sorted(list(set(predict_score.flatten()))))  # Get unique predicted scores.
    thresholds = sorted_predict_score[np.int32(len(sorted_predict_score) * np.arange(1, 1000) / 1000)]  # Create thresholds.
    thresholds = np.mat(thresholds)  # Convert thresholds to matrix.
    thresholds_num = thresholds.shape[1]  # Get number of thresholds.

    predict_score_matrix = np.tile(predict_score, (thresholds_num, 1))  # Create matrix of predictions.
    negative_index = np.where(predict_score_matrix < thresholds.T)  # Find indices for negative predictions.
    positive_index = np.where(predict_score_matrix >= thresholds.T)  # Find indices for positive predictions.
    predict_score_matrix[negative_index] = 0  # Set negative predictions to 0.
    predict_score_matrix[positive_index] = 1  # Set positive predictions to 1.
    TP = predict_score_matrix.dot(real_score.T)  # Calculate True Positives.
    FP = predict_score_matrix.sum(axis=1) - TP  # Calculate False Positives.
    FN = real_score.sum() - TP  # Calculate False Negatives.
    TN = len(real_score.T) - TP - FP - FN  # Calculate True Negatives.

    fpr = FP / (FP + TN)  # Calculate False Positive Rate.
    tpr = TP / (TP + FN)  # Calculate True Positive Rate.
    ROC_dot_matrix = np.mat(sorted(np.column_stack((fpr, tpr)).tolist())).T  # Create ROC curve data_DC.
    ROC_dot_matrix.T[0] = [0, 0]  # Include point (0,0) in ROC.
    ROC_dot_matrix = np.c_[ROC_dot_matrix, [1, 1]]  # Include point (1,1) in ROC.

    x_ROC = ROC_dot_matrix[0].T  # Get x-coordinates for ROC.
    y_ROC = ROC_dot_matrix[1].T  # Get y-coordinates for ROC.
    auc = 0.5 * (x_ROC[1:] - x_ROC[:-1]).T * (y_ROC[:-1] + y_ROC[1:])  # Calculate AUC.

    recall_list = tpr  # Recall values.
    precision_list = TP / (TP + FP)  # Precision values.
    PR_dot_matrix = np.mat(sorted(np.column_stack((recall_list, precision_list)).tolist())).T  # Create PR curve data_DC.
    PR_dot_matrix.T[0] = [0, 1]  # Include point (0,1) in PR.
    PR_dot_matrix = np.c_[PR_dot_matrix, [1, 0]]  # Include point (1,0) in PR.

    x_PR = PR_dot_matrix[0].T  # Get x-coordinates for PR.
    y_PR = PR_dot_matrix[1].T  # Get y-coordinates for PR.
    aupr = 0.5 * (x_PR[1:] - x_PR[:-1]).T * (y_PR[:-1] + y_PR[1:])  # Calculate AUPR.

    f1_score_list = 2 * TP / (len(real_score.T) + TP - TN)  # Calculate F1 score.
    accuracy_list = (TP + TN) / len(real_score.T)  # Calculate accuracy.
    specificity_list = TN / (TN + FP)  # Calculate specificity.

    max_index = np.argmax(f1_score_list)  # Find index of max F1 score.
    f1_score = f1_score_list[max_index]  # Get max F1 score.
    accuracy = accuracy_list[max_index]  # Get accuracy for max F1 score.
    specificity = specificity_list[max_index]  # Get specificity for max F1 score.
    recall = recall_list[max_index]  # Get recall for max F1 score.
    precision = precision_list[max_index]  # Get precision for max F1 score.

    print(f' auc:{auc[0, 0]:.4f}, aupr:{aupr[0, 0]:.4f}, f1_score:{f1_score:.4f}, accuracy:{accuracy:.4f}, recall:{recall:.4f}, specificity:{specificity:.4f}, precision:{precision:.4f}')  # Output metrics.
    return [real_score, predict_score, auc[0, 0], aupr[0, 0], f1_score, accuracy, recall, specificity, precision], \
           ['y_true', 'y_score', 'auc', 'prc', 'f1_score', 'acc', 'recall', 'specificity', 'precision']  # Return metrics and names.

def mult_func(num_layers):
    """
    Function to train models with grid search and save the results.
    """
    all_list = []  # List to store all results.

    selector = ModelSelector()  # Create ModelSelector instance.
    models = selector.get_models([])  # Get all models.
    results_dict = selector.train_with_grid_search(models)  # Train models.
    all_list.append(results_dict)  # Store results.

    save_path = "results.npy"  # Path to save results.
    np.save(save_path, all_list)  # Save results to file.

    print(all_list)  # Print all results.
    return results_dict  # Return results.

if __name__ == '__main__':
    range_list = range(1, 2)  # Define range for multiprocessing (only 1 in this case).
    mut = True  # Flag for multiprocessing.
    if mut:
        multiprocessing.set_start_method('spawn')  # Set start method for multiprocessing.

        pool = multiprocessing.Pool(processes=min(len(range_list), multiprocessing.cpu_count() - 1))  # Create a pool of workers.

        items_to_process = range_list  # Define items to process.

        pool.map(mult_func, items_to_process)  # Map function to items for processing.

        pool.close()  # Close the pool.
        pool.join()  # Wait for all processes to finish.
