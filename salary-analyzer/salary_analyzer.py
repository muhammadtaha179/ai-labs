import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression

# Dataset
df = pd.DataFrame({
    "Employee": ["Asad", "Mudassir", "Talha", "Taha"],
    "Salary": [90000, 80000, 70000, 60000],
    "Hours_Worked": [5, 4, 3, 2]
})

# Statistics
print("Mean Salary:", np.mean(df["Salary"]))
print("Median Salary:", np.median(df["Salary"]))
mode_result = stats.mode(df["Salary"], keepdims=False)
print("Mode Salary:", mode_result.mode)

# Visualization
sns.barplot(x="Hours_Worked", y="Salary", data=df)
plt.title("Employees Salary Analyzer")
plt.show()

# Machine Learning
X = df[["Hours_Worked"]]
y = df[["Salary"]]
model = LinearRegression()
model.fit(X, y)

predicted = model.predict(pd.DataFrame({"Hours_Worked": [12]}))
print("Predicted Salary for 12 hours:", predicted)
