# Customer Churn Prediction using Artificial Neural Network

## Overview
This project predicts customer churn for a telecom company using an Artificial Neural Network (ANN). Customer churn refers to the rate at which customers stop doing business with a company. Predicting churn helps businesses retain customers and reduce losses.

The project covers the entire workflow:
- Data exploration and visualization
- Data preprocessing
- Building and training an ANN model
- Model evaluation using metrics and confusion matrix

## Dataset
The dataset used in this project is the **Telco Customer Churn Dataset**, obtained from Kaggle. It contains 7043 rows and 21 columns with customer information and churn status.

**Independent variables (features)**:
- `gender`, `SeniorCitizen`, `Partner`, `Dependents`, `tenure`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`

**Dependent variable (target)**:
- `Churn` (1 indicates churn, 0 indicates retained customer)

## Requirements
- Python 3.8+
- Jupyter Notebook or Google Colab
- Libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `tensorflow`, `keras`

## Steps Included in the Notebook
1. **Import Libraries** – Required libraries for data processing, visualization, and modeling.
2. **Load Dataset** – Read CSV file into a Pandas DataFrame.
3. **Data Exploration and Visualization** – Check null values, data types, distribution of features, and target variable.
4. **Data Preprocessing**:
   - Drop unnecessary columns (`customerID`)
   - Handle missing values
   - Convert categorical variables to numeric
   - Scale numerical features
5. **Split Data** – Divide data into training and testing sets.
6. **Build ANN Model** – Define input, hidden, and output layers.
7. **Compile and Train Model** – Use `adam` optimizer and `binary_crossentropy` loss function.
8. **Evaluate Model** – Check accuracy, classification report, and confusion matrix.
9. **Visualize Results** – Compare original vs predicted churn and visualize metrics.

## How to Run
1. Clone this repository.
2. Ensure the `churn.csv` dataset is in the same directory as the notebook.
3. Open the `Customer_Churn_ANN.ipynb` in Jupyter Notebook or Google Colab.
4. Run all cells sequentially to reproduce results.

## Results
- Confusion matrix shows true positives, true negatives, false positives, and false negatives.
- Classification report provides precision, recall, and F1-score for the model.
- The model predicts churn with high accuracy after preprocessing and training.

## References
- Kaggle Telco Customer Churn Dataset: [https://www.kaggle.com/blastchar/telco-customer-churn](https://www.kaggle.com/blastchar/telco-customer-churn)
- Kaggle Notebook: [https://www.kaggle.com/code/sanhithreddy/telco-customer-churn-prediction-using-ann](https://www.kaggle.com/code/sanhithreddy/telco-customer-churn-prediction-using-ann)
- TensorFlow & Keras Documentation: [https://www.tensorflow.org/](https://www.tensorflow.org/)
- Analytics Vidhya Article: Customer Churn Prediction using ANN

## Author
**Thikkavarapu Sanhith**  
LinkedIn: [https://www.linkedin.com/in/sanhith30/](https://www.linkedin.com/in/sanhith30)
