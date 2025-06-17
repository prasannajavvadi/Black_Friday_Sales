# Black-Friday-Sales-Prediction
![alt text](https://searchengineland.com/figz/wp-content/seloads/2014/12/black-friday1-ss-1920.jpg "Black Friday Sales Prediction")

## Table Of Contents
  - [Project Introduction](#project-introduction)
  - [Dataset Description](#dataset-description)
  - [EDA](#eda)
  - [Data Preprocessing](#data-preparation)
  - [Modeling Phase](#modeling-phase)
  - [Evaluation Metric](#evaluation-metric)
  - [Conclusion](#conclusion)

### Project Introduction
Black Friday is an informal name for the Friday following Thanksgiving Day in the United States, which is celebrated on the fourth Thursday of November. The day after Thanksgiving has been regarded as the beginning of the United States Christmas shopping season since 1952, although the term "Black Friday" did not become widely used until more recent decades. Many stores offer highly promoted sales on Black Friday and open very early, such as at midnight, or may even start their sales at some time on Thanksgiving. The major challenge for a Retail store or eCommerce business is to choose product price such that they get maximum profit at the end of the sales. Our project deals with determining the product prices based on the historical retail store sales data. After generating the predictions, our model will help the retail store to decide the price of the products to earn more profits.
Sure, here's a detailed README file for your project to upload on GitHub:

---

# Black Friday Sales Prediction

![Black Friday Sales](https://example.com/black_friday_sales_image.png)

## Overview

This project aims to predict customer purchase amounts during Black Friday sales using various machine learning techniques. The dataset used for this project is obtained from Kaggle and contains information about purchases made by customers during the Black Friday sales.

## Dataset

The dataset used in this project is the Black Friday Sales Dataset from Kaggle. It includes various features such as user demographics, product information, and purchase amounts.

### Dataset Details

- **User_ID**: Unique identifier for each user.
- **Product_ID**: Unique identifier for each product.
- **Gender**: Gender of the user.
- **Age**: Age group of the user.
- **Occupation**: Occupation code of the user.
- **City_Category**: Category of the city.
- **Stay_In_Current_City_Years**: Number of years the user has stayed in the current city.
- **Marital_Status**: Marital status of the user.
- **Product_Category_1**: Product category (1).
- **Product_Category_2**: Product category (2).
- **Product_Category_3**: Product category (3).
- **Purchase**: Purchase amount.

## Project Structure

- `data/`: Contains the dataset file `BlackFridaySales.csv`.
- `notebooks/`: Jupyter notebooks with detailed exploratory data analysis (EDA) and model training steps.
- `src/`: Source code for data preprocessing, model training, and evaluation.
- `models/`: Saved models and pipelines.
- `README.md`: Project overview and instructions.
- `requirements.txt`: List of required Python packages.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/black-friday-sales-prediction.git
   cd black-friday-sales-prediction
   ```

2. Create a virtual environment and activate it:

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Data Preprocessing

The first step is to preprocess the data. This includes handling missing values, encoding categorical variables, and splitting the data into training and testing sets.

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load the dataset
data = pd.read_csv("https://raw.githubusercontent.com/nanthasnk/Black-Friday-Sales-Prediction/master/Data/BlackFridaySales.csv")

# Handle missing values
data['Product_Category_2'] = data['Product_Category_2'].fillna(0).astype('int64')
data['Product_Category_3'] = data['Product_Category_3'].fillna(0).astype('int64')

# Encode categorical variables
le = LabelEncoder()
data['Gender'] = le.fit_transform(data['Gender'])
data['Age'] = le.fit_transform(data['Age'])
data['City_Category'] = le.fit_transform(data['City_Category'])

# Drop unnecessary columns
data = data.drop(["User_ID", "Product_ID"], axis=1)

# Split the data
X = data.drop("Purchase", axis=1)
y = data['Purchase']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)
```

### Model Training

We use various machine learning models for training and evaluation, including Linear Regression, Decision Tree Regressor, Random Forest Regressor, and XGBoost Regressor.

```python
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost.sklearn import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from math import sqrt

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_y_pred = lr.predict(X_test)
print("Linear Regression RMSE: ", sqrt(mean_squared_error(y_test, lr_y_pred)))

# Decision Tree Regressor
dt_regressor = DecisionTreeRegressor(random_state=0)
dt_regressor.fit(X_train, y_train)
dt_y_pred = dt_regressor.predict(X_test)
print("Decision Tree RMSE: ", sqrt(mean_squared_error(y_test, dt_y_pred)))

# Random Forest Regressor
rf_regressor = RandomForestRegressor(random_state=0)
rf_regressor.fit(X_train, y_train)
rf_y_pred = rf_regressor.predict(X_test)
print("Random Forest RMSE: ", sqrt(mean_squared_error(y_test, rf_y_pred)))

# XGBoost Regressor
xgb_reg = XGBRegressor(learning_rate=1.0, max_depth=6, min_child_weight=40, seed=0)
xgb_reg.fit(X_train, y_train)
xgb_y_pred = xgb_reg.predict(X_test)
print("XGBoost RMSE: ", sqrt(mean_squared_error(y_test, xgb_y_pred)))
```

### Evaluation

Evaluate the models using metrics such as Mean Absolute Error, Mean Squared Error, and R-Squared.

```python
print("Linear Regression MAE: ", mean_absolute_error(y_test, lr_y_pred))
print("Linear Regression MSE: ", mean_squared_error(y_test, lr_y_pred))
print("Linear Regression R2: ", r2_score(y_test, lr_y_pred))

print("Decision Tree MAE: ", mean_absolute_error(y_test, dt_y_pred))
print("Decision Tree MSE: ", mean_squared_error(y_test, dt_y_pred))
print("Decision Tree R2: ", r2_score(y_test, dt_y_pred))

print("Random Forest MAE: ", mean_absolute_error(y_test, rf_y_pred))
print("Random Forest MSE: ", mean_squared_error(y_test, rf_y_pred))
print("Random Forest R2: ", r2_score(y_test, rf_y_pred))

print("XGBoost MAE: ", mean_absolute_error(y_test, xgb_y_pred))
print("XGBoost MSE: ", mean_squared_error(y_test, xgb_y_pred))
print("XGBoost R2: ", r2_score(y_test, xgb_y_pred))
```

## Results

- **Linear Regression RMSE**: `XXXX`
- **Decision Tree RMSE**: `XXXX`
- **Random Forest RMSE**: `XXXX`
- **XGBoost RMSE**: `XXXX`

The XGBoost model showed the best performance in terms of RMSE, indicating its superior ability to predict purchase amounts.

## Conclusion

This project demonstrates the potential of machine learning models in predicting customer purchase amounts during Black Friday sales. Future work could include hyperparameter tuning, trying out other machine learning algorithms, and further optimizing the models.

## Contributing

If you wish to contribute to this project, please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a new Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The Kaggle community for providing the dataset.
- All contributors to this project.

---

Feel free to modify and customize this README as per your specific needs and additional details.