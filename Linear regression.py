from Helper.AI_Helper import *
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error

Iris = Get_Dataset("himanshunakrani/iris-dataset", "iris.csv")


y = Iris["sepal_length"]
x = Iris.drop(["species","sepal_length"], axis=1)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
rmse = root_mean_squared_error(y_test, y_pred)
# shows the differnce in the measured units
print(rmse)
