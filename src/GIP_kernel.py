import numpy as np
import pandas as pd


def GIP_kernel(Drug_Target_Assoc):
    def calculate_r(Drug_Target_Assoc):
        # Calculate the r value used in the GIP kernel formula
        num_rows = Drug_Target_Assoc.shape[0]
        summation = 0
        for i in range(num_rows):
            row_norm = np.linalg.norm(Drug_Target_Assoc[i, :])
            row_norm_squared = np.square(row_norm)
            summation += row_norm_squared
        r_value = summation / num_rows
        return r_value

    # Number of rows (drugs or targets)
    num_rows = Drug_Target_Assoc.shape[0]
    # Initialize a matrix to store the GIP kernel results
    kernel_matrix = np.zeros((num_rows, num_rows))

    # Calculate the denominator in the GIP formula
    r_value = calculate_r(Drug_Target_Assoc)

    # Calculate the GIP kernel similarity matrix
    for i in range(num_rows):
        for j in range(num_rows):
            # Calculate the numerator in the GIP formula
            diff_norm_squared = np.square(np.linalg.norm(Drug_Target_Assoc[i, :] - Drug_Target_Assoc[j, :]))
            if r_value == 0:
                kernel_matrix[i][j] = 0
            elif i == j:
                kernel_matrix[i][j] = 1
            else:
                kernel_matrix[i][j] = np.exp(-diff_norm_squared / r_value)

    return kernel_matrix


if __name__ == '__main__':
    # Use relative path ../data_DC/ for reading and saving data
    drug_target_matrix = np.array(pd.read_csv('../data_DC/d_t.csv', header=None, index_col=None))

    # Calculate GIP similarity for drugs and targets
    GIP_drug_sim = GIP_kernel(drug_target_matrix)
    GIP_target_sim = GIP_kernel(drug_target_matrix.T)

    # Save the results to ../data_DC/ directory
    pd.DataFrame(GIP_drug_sim).to_csv('../data_DC/d_gs.csv', header=None, index=None, float_format='%.10f')
    pd.DataFrame(GIP_target_sim).to_csv('../data_DC/t_gs.csv', header=None, index=None, float_format='%.10f')
