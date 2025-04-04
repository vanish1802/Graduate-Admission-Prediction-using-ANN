# Graduate Admission Prediction Using ANN

## **Overview**
This project demonstrates the use of an Artificial Neural Network (ANN) to predict graduate admission chances based on various academic and profile-related features. The dataset contains information about GRE scores, TOEFL scores, CGPA, and other factors that influence admission chances. The model is implemented using TensorFlow and Keras.

---

## **Dataset**
- **Source**: [Graduate Admissions Dataset](https://www.kaggle.com/mohansacharya/graduate-admissions)
- **Features**:
  - GRE Score
  - TOEFL Score
  - University Rating
  - Statement of Purpose (SOP)
  - Letter of Recommendation (LOR)
  - CGPA
  - Research Experience (0 or 1)
- **Target**: Chance of Admission (a continuous value between 0 and 1)

---

## **Project Workflow**
1. **Data Loading and Preprocessing**
   - The dataset is loaded and cleaned by dropping irrelevant columns (e.g., `Serial No.`).
   - Features are scaled using Min-Max Scaling to normalize the data.

2. **Model Architecture**
   - A Sequential model is built with the following layers:
     - `Dense (7 neurons)`: Fully connected layer with ReLU activation for input features.
     - `Dense (7 neurons)`: Fully connected layer with ReLU activation for intermediate processing.
     - `Dense (1 neuron)`: Output layer with linear activation for regression.

3. **Compilation**
   - **Loss Function**: Mean Squared Error (MSE)
   - **Optimizer**: Adam
   - **Metrics**: Accuracy

4. **Model Training**
   - The model is trained for 100 epochs with a validation split of 20%.

5. **Evaluation**
   - The model's performance is evaluated using the R² score on the test set.

6. **Visualization**
   - Training and validation loss are plotted to analyze model performance over epochs.

---

## **Results**
- **R² Score**: 0.7727  
The model explains approximately 77.27% of the variance in admission chances based on the input features.

---

## **Model Summary**

| Layer (Type)       | Output Shape | Parameters |
|--------------------|--------------|------------|
| Dense (ReLU)       | (None, 7)    | 56         |
| Dense (ReLU)       | (None, 7)    | 56         |
| Dense (Linear)     | (None, 1)    | 8          |

- Total Parameters: `120`
- Trainable Parameters: `120`

---

## **Conclusion**
This project demonstrates how an ANN can effectively predict graduate admission chances based on academic and profile-related features. While the R² score indicates good performance, further improvements could be made by experimenting with additional features, hyperparameter tuning, or more complex architectures.

---

### How to Use This Repository
1. Clone this repository:
   ```bash
   git clone https://github.com/your_username/graduate-admission-prediction-ann.git
   ```
2. Navigate to the directory:
   ```bash
   cd graduate-admission-prediction-ann
   ```
3. Open the Jupyter Notebook:
   ```bash
   jupyter notebook graduate-admission-prediction-using-ann.ipynb
   ```
4. Execute the cells in the notebook to train and evaluate the model.

---

### Dependencies
To run this project, install the following Python libraries:
- TensorFlow
- NumPy
- Pandas
- Matplotlib
- scikit-learn

Install them using:
```bash
pip install tensorflow numpy pandas matplotlib scikit-learn
```

---

### Repository Contents
- `graduate-admission-prediction-using-ann.ipynb`: Jupyter Notebook containing the implementation.
- `README.md`: Project documentation.

---

### License
This project is licensed under the MIT License. Feel free to use and modify it as needed!

Citations:
[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/51708891/516fb3f0-26b3-4896-8cfc-b8e9a66950fc/graduate-admission-prediction-using-ann.ipynb

---
