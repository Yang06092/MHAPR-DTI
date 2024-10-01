# My Paper Title  
**DTI-MHAPR: Optimized Drug-Target Interaction Prediction via PCA-Enhanced Features and Heterogeneous Graph Attention Networks**  

## Requirements  
To install the requirements, follow these steps:

1. Ensure you are using Python 3.8.  
2. Run the following command to install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Data Preparation  
3. Download the dataset and extract the contents of the `.rar` file directly into the current folder.

4. **Simulated Data**:
   - Inside the `tests` folder, there is a script for generating simulated data using random functions.
   - The script will generate the following datasets:
     - `d_t.csv`: A randomly generated M×N binary matrix (0s and 1s), representing known drug-target interactions (DTIs).
     - `d_ss.csv`: A randomly generated M×M matrix with float values between 0 and 1, representing drug sequence similarity.
     - `p_ss.csv`: A randomly generated N×N matrix with float values between 0 and 1, representing target structure similarity.
   - These generated datasets will be saved in the `data_DC` folder.

5. **Gaussian Kernel Similarity Matrices**:
   - After generating the initial simulated data, you can run the `GIP_kernel` function to generate Gaussian kernel similarity matrices:
     - `d_gs.csv`: The Gaussian kernel similarity matrix for drugs.
     - `t_gs.csv`: The Gaussian kernel similarity matrix for targets.
   - Both matrices will be saved in the `data_DC` folder.

## Feature Extraction & Training  
6. To train the model and extract features as described in the paper, execute the following command:
    ```bash
    python main.py
    ```
   - Running `main.py` will generate a `mid_data` folder. Inside this folder, you'll find the embedded features extracted during the 5-fold cross-validation process.
   - The cross-validation results for the feature extraction phase will be saved in the `save_file` folder.

## Prediction  
7. After running `main.py`, execute the following command to obtain final predictions using the trained model on the extracted features:
    ```bash
    python ./models/ML_model.py
    ```
   - This will invoke the PCA model for complete feature optimization and then use the Random Forest algorithm for DTI prediction.

## Baseline  
- CGHCN: [GitHub Link](https://github.com/LiangXujun/CGHCN)  
- MINIMDA: [GitHub Link](https://github.com/chengxu123/MINIMDA)  
- GATECDA: [GitHub Link](https://github.com/yjslzx/GATECDA)  
- MNGACDA: [GitHub Link](https://github.com/youngbo9i/MNGACDA)  
- DTI-CNN: [GitHub Link](https://github.com/MedicineBiology-AI/DTI-CNN)  
- FSI_framework: [GitHub Link](https://github.com/piyanuttnk/FSI_framework)  
- HFHLMDA: [GitHub Link](https://github.com/LiangXujun/CGHCN/HFHLMDA_main.py)

## Thanks!
