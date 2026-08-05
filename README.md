# NeuroFive ML Track

Track: Machine Learning / Data Science
Intern: Rubab Zulfiqar

## Task 1: Environment Setup + Titanic EDA

**Objective:** Set up the Python data science toolkit and perform an initial Exploratory Data Analysis (EDA) on the Titanic dataset before any modeling.

**What I did:**
- Set up Python environment with pandas and NumPy in VS Code
- Loaded the Titanic dataset (`train.csv`) using `pandas.read_csv()`
- Inspected the dataset using `.info()`, `.describe()`, and `.head()`
- Identified missing values across columns (Age, Cabin, Embarked)
- Classified columns into categorical vs. numerical features
- Summarized findings in a markdown "data story" cell

**Key Findings:**
- Dataset: 891 rows, 12 columns
- Missing values: `Age` (~177), `Cabin` (~687), `Embarked` (2)
- Numerical columns: Age, Fare, SibSp, Parch, Pclass
- Categorical columns: Sex, Embarked, Ticket, Cabin, Name

**Tools used:** Python, VS Code, pandas, NumPy

**Notebook:** [`titanic_eda.ipynb`](./titanic_eda.ipynb)

## Task 2: Data Cleaning + Visualization

**Objective:** Clean the Titanic dataset and visualize patterns to understand the data before modeling.

**What I did:**
- Handled missing values: `Age` filled with median, `Embarked` filled with mode, `Cabin` dropped (too many missing)
- Detected outliers in `Fare` using a boxplot
- Created 4 visualizations: histogram (Age), boxplot (Fare), bar chart (Survival by Pclass), correlation heatmap
- Concluded that `Pclass` most strongly affects survival

**Tools used:** Python, pandas, matplotlib, seaborn

## Task 3: Logistic Regression - Survival Prediction

**Objective:** Build the first machine learning classification model to predict passenger survival.

**What I did:**
- Encoded categorical columns (`Sex`, `Embarked`) using `pd.get_dummies()`
- Split the dataset into training and test sets (80/20 split) using `train_test_split`
- Trained a Logistic Regression model on features: Pclass, Age, SibSp, Parch, Fare, Sex, Embarked
- Evaluated the model using accuracy score and a confusion matrix

**Result:** Achieved an accuracy of **0.81** on the test set.

**Tools used:** Python, scikit-learn, pandas

## Task 4: Linear Regression - House Price Prediction

**Objective:** Predict house prices using regression, a core ML technique for pricing and forecasting problems.

**What I did:**
- Used the California Housing dataset from `sklearn.datasets`
- Selected 5 features: MedInc, HouseAge, AveRooms, AveBedrms, Population
- Trained a Linear Regression model with an 80/20 train-test split
- Evaluated using RMSE and R² score
- Plotted predicted vs. actual prices to visually assess model quality

**Result:** RMSE = **0.80**, R² Score = **0.51**

**Tools used:** Python, scikit-learn, pandas, matplotlib

## Task 5: Model Evaluation + Hyperparameter Tuning

**Objective:** Go beyond accuracy to properly evaluate a classification model, and tune hyperparameters systematically.

**What I did:**
- Calculated Precision, Recall, and F1-score using `classification_report`
- Explained why accuracy alone is misleading for imbalanced datasets
- Tuned the Logistic Regression model's `C` and `solver` using `GridSearchCV`
- Compared original vs tuned model performance

**Result:** Original model performed slightly better (Accuracy 0.81) than the tuned model (Accuracy 0.78) — tuning didn't improve results in this case, showing that default parameters aren't always beaten by tuning.

**Tools used:** Python, scikit-learn

## Task 6: Customer Churn Prediction (Decision Tree vs Logistic Regression)

**Objective:** Predict customer churn using the Telco Customer Churn dataset — a real business problem across telecom, banking, and SaaS.

**What I did:**
- Performed EDA on churn vs contract type, tenure, and monthly charges
- Noted class imbalance (~73% No churn, ~27% Yes churn) and used stratified train-test split
- Trained and compared Decision Tree and Logistic Regression models
- Identified top 3 churn drivers using `.feature_importances_`

**Results:**
- Decision Tree Accuracy: 0.79
- Logistic Regression Accuracy: 0.80 (slightly better, especially on churn class)
- Top churn drivers: **tenure**, **Fiber Optic internet service**, **TotalCharges**

**Tools used:** Python, scikit-learn, pandas, matplotlib, seaborn