[5-lineaer_regression.py](https://github.com/user-attachments/files/34319928/9-lineaer_regression.py)
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


#load data
df = pd.read_csv("Hours_marks.csv")

#split in x and y
X = df[["Hours"]]   # Feature (independent)
y = df["Marks"]     # Target (dependent)


# split into 4 part
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,      # 20% for testing
    random_state=42     # reproducible split
)


# Create model
model = LinearRegression()


# train the model
model.fit(X_train, y_train)

# predict on test data
y_hat = model.predict(X_test)



# evaluate model
mse = mean_squared_error(y_test, y_hat)
print(mse)



# Predict marks for a student who studied 11 hours
new_hours = [[11]]
predicted_marks = model.predict(new_hours)

print("Predicted marks for 11 hours of study: ", predicted_marks)


