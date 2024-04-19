import pandas as pd
from sklearn.datasets import load_diabetes

X, y = load_diabetes(return_X_y=True, as_frame=True)
X.head()
y.head()

# 10 baseline variables (features):
# age - age in years
# sex - male or female
# bmi - body mass index
# bp - average blood pressure
# s1 - TC: total serum cholesterol
# s2 - LDL: low-density lipoproteins
# s3 - HDL: high-density lipoproteins
# s4 - TCH: total cholesterol / HDL
# s5 - LTG: possibly log of serum triglycerides level
# s6 - GLU: blood sugar level

# One target variable: a quantitative measure of disease progression one year after baseline


# Each of the 10 feature variables have been mean centered and scaled by the standard deviation times the square root of the number of sample (ie the sum of squares of each column totals 1).