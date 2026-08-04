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