# Iris-Classification-Using-Decision-Tree-and-Random-Forest
# Iris Classification Using Decision Tree and Random Forest

## 📌 Project Overview
This project implements classification models using Decision Tree and Random Forest on the famous Iris dataset. The objective is to classify iris flowers into three species based on their sepal and petal dimensions.

## 📂 Dataset
- **Source**: Iris dataset (CSV format)
- **Features**:
  - `SepalLengthCm`
  - `SepalWidthCm`
  - `PetalLengthCm`
  - `PetalWidthCm`
- **Target**: `Species` (Setosa, Versicolor, Virginica)

## 🚀 Preprocessing Steps
1. **Remove Duplicates**: Ensure data integrity.
2. **Feature Scaling**: Apply Min-Max Scaling to normalize numerical features.
3. **Outlier Detection**: Remove outliers in `SepalWidthCm` using the IQR method.
4. **One-Hot Encoding**: Convert categorical target variable (`Species`) into numerical format.

## 📊 Model Implementation
### Decision Tree Classifier
- Trained with **max depth = 3** for better generalization.
- Evaluated on training and testing datasets using:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
- Cross-validation applied to validate performance.

### Random Forest Classifier
- Implemented for better generalization.
- Trained with **100 estimators**.
- Achieved improved accuracy on the test dataset.

## 🔍 Results
### Decision Tree Performance:
| Metric       | Training Score | Testing Score |
|-------------|---------------|--------------|
| Accuracy    | **X.XXXX**     | **X.XXXX**   |
| Precision   | **X.XXXX**     | **X.XXXX**   |
| Recall      | **X.XXXX**     | **X.XXXX**   |
| F1 Score    | **X.XXXX**     | **X.XXXX**   |
| Cross-Validation Accuracy | **X.XXXX** |

### Random Forest Test Accuracy:
- **X.XXXX**

### Feature Importance (Decision Tree)
| Feature        | Importance |
|---------------|------------|
| PetalLengthCm | **X.XX**   |
| PetalWidthCm  | **X.XX**   |
| SepalLengthCm | **X.XX**   |
| SepalWidthCm  | **X.XX**   |

## 📌 Conclusion
- The **Random Forest model** achieved better accuracy compared to the Decision Tree model.
- Feature importance analysis highlights that **petal dimensions** are the most influential in classification.
- Further improvements can be made using hyperparameter tuning and ensemble learning.

## ⚡ How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo-name.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the script:
   ```bash
   python iris_classification.py
   ```

## 📬 Contributing
Feel free to fork this repository and improve the model by experimenting with hyperparameter tuning, feature engineering, and alternative classifiers.

## 📜 License
This project is open-source and available under the MIT License.

