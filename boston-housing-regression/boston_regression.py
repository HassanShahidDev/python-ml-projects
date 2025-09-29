# boston_regression.py
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

boston = load_boston()
X_train, X_test, y_train, y_test = train_test_split(
    boston.data, boston.target, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Predicted:", y_pred)
print("Actual   :", y_test)
print("MSE      :", mean_squared_error(y_test, y_pred))


#run
#python boston-housing-regression/boston_regression.py
