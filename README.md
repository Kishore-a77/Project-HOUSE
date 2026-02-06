# House Price Prediction using Linear Regression

A machine learning project that predicts house sale prices using Linear Regression based on various property features and characteristics.

## 📋 Overview

This project implements a supervised learning model to predict residential property prices. By analyzing multiple housing features such as location, size, quality, and amenities, the model provides accurate price estimations that can assist buyers, sellers, and real estate professionals in making informed decisions.

## ✨ Features

- **Comprehensive Data Analysis**: Exploratory data analysis (EDA) with correlation analysis and feature distributions
- **Data Preprocessing**: Complete pipeline including handling missing values, feature encoding, and data cleaning
- **Feature Engineering**: Label encoding for categorical variables and feature selection
- **Linear Regression Model**: Implementation of Linear Regression for price prediction
- **Model Evaluation**: Performance metrics including Mean Absolute Percentage Error (MAPE)
- **Rich Visualizations**: Correlation heatmaps, distribution plots, and categorical feature analysis

## 🚀 Getting Started

### Prerequisites

Ensure you have Python 3.7+ installed along with the following libraries:

- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- openpyxl (for Excel file handling)

### Installation

Install required packages using pip:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn openpyxl
```

### Dataset

The project uses `HousePricePrediction.xlsx`, which contains various housing features including:

**Property Features:**
- Location and neighborhood characteristics
- House size and dimensions
- Property condition and quality ratings
- Year built and remodeling information
- Basement, garage, and other amenities
- **Target Variable**: SalePrice (house selling price)

## 📁 Project Structure

```
.
├── House_Price_Prediction.ipynb    # Main Jupyter notebook with implementation
├── HousePricePrediction.xlsx       # Dataset file
└── README.md                        # Project documentation
```

## 🔧 Usage

1. **Clone or download** this repository
2. **Ensure the dataset** `HousePricePrediction.xlsx` is in the appropriate directory
3. **Open** the Jupyter notebook:
   ```bash
   jupyter notebook House_Price_Prediction.ipynb
   ```
4. **Update the filepath** in the first cell to match your dataset location:
   ```python
   filepath = r'path/to/your/HousePricePrediction.xlsx'
   ```
5. **Execute cells sequentially** to run the complete pipeline

## 📊 Methodology

### 1. Data Loading and Exploration
- Load the housing dataset from Excel file
- Display initial data samples
- Analyze dataset shape and structure
- Identify categorical, integer, and float variables

### 2. Exploratory Data Analysis (EDA)
- **Correlation Analysis**: Generate correlation matrix heatmap for numeric features
- **Categorical Features**: Analyze unique values and distributions
- **Distribution Plots**: Visualize categorical feature distributions across 11x4 subplots

### 3. Data Preprocessing
- **Feature Cleaning**: Remove unnecessary columns (Id)
- **Missing Value Treatment**: 
  - Fill missing SalePrice values with mean
  - Drop remaining rows with missing values
- **Verification**: Confirm no missing values remain in the dataset

### 4. Feature Engineering
- **Label Encoding**: Convert categorical variables to numerical format using LabelEncoder
- **Feature Transformation**: Process all object-type columns for model compatibility

### 5. Model Training
- **Train-Test Split**: 80% training, 20% validation split
- **Model Selection**: Linear Regression algorithm
- **Features (X)**: All columns except SalePrice
- **Target (Y)**: SalePrice column

### 6. Model Evaluation
- **Prediction**: Generate price predictions on validation set
- **Performance Metric**: Mean Absolute Percentage Error (MAPE)
- **Results**: Evaluate model accuracy and prediction quality

## 📈 Results

The Linear Regression model successfully predicts house prices with the following characteristics:

- **Algorithm**: Linear Regression
- **Training Split**: 80-20 train-validation split
- **Evaluation Metric**: Mean Absolute Percentage Error (MAPE)
- **Features**: Multiple property characteristics including location, size, quality, and amenities

### Key Insights

- Correlation analysis reveals relationships between different property features and sale price
- Categorical features show varied distributions across different property types
- The model provides a baseline for house price prediction that can be further enhanced

## 🔮 Future Enhancements

Potential improvements for the project:

- **Advanced Models**: Implement Random Forest, XGBoost, or Neural Networks for better accuracy
- **Feature Selection**: Use techniques like Recursive Feature Elimination (RFE) or feature importance
- **Hyperparameter Tuning**: Optimize model parameters using GridSearchCV or RandomizedSearchCV
- **Cross-Validation**: Implement k-fold cross-validation for robust performance estimation
- **Additional Metrics**: Include R², RMSE, and MAE for comprehensive evaluation
- **Outlier Detection**: Identify and handle outliers in the dataset
- **Feature Scaling**: Apply StandardScaler or MinMaxScaler for normalized features
- **Deployment**: Create a web application for real-time price predictions

## 🛠️ Technologies Used

- **Python**: Core programming language
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **Matplotlib & Seaborn**: Data visualization
- **Scikit-learn**: Machine learning algorithms and preprocessing
- **Jupyter Notebook**: Interactive development environment

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is open-source and available under the MIT License.

## 📧 Contact

For questions, suggestions, or feedback, please open an issue in the repository.

## 🎓 Acknowledgments

This project demonstrates:
- Supervised learning techniques for regression problems
- Real estate price prediction methodology
- Data preprocessing and feature engineering best practices
- Model evaluation and validation strategies

---

**Note**: This is an educational project showcasing machine learning applications in real estate. The model and techniques can be adapted and enhanced for production use cases with additional feature engineering and advanced algorithms.
3. **Regularization**: Try Ridge or Lasso regression to prevent overfitting
4. **Ensemble Methods**: Combine multiple models for improved predictions
5. **Domain Knowledge**: Incorporate real estate expertise for better feature selection
