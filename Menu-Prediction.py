import pandas as pd

# Load the data into a pandas DataFrame
data = pd.DataFrame([['burger', 'Monday', 10, 20, 4, 50],
                    ['burger', 'Tuesday', 12, 22, 4, 55],
                    ['burger', 'Wednesday', 14, 24, 3, 45],
                    ['fries', 'Monday', 8, 18, 5, 60],
                    ['fries', 'Tuesday', 10, 20, 4, 55],
                    ['fries', 'Wednesday', 12, 22, 3, 45],
                    ['pizza', 'Monday', 12, 22, 5, 65],
                    ['pizza', 'Tuesday', 14, 24, 4, 60],
                    ['pizza', 'Wednesday', 16, 26, 3, 50]],
                   columns=['menu_item', 'day_of_sale', 'wind_speed', 'temperature', 'menu_score', 'kilos_sold'])

# Create a list of features to use in the model
features = ["wind_speed", "temperature", "menu_score"]

# Define the target variable
target = "kilos_sold"

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.2)

# Create a Linear Regression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Use the model to predict the kilos sold for the next year
next_year_predictions = model.predict(X_test)

# Get the index of the menu item with the highest predicted sales
best_menu_index = next_year_predictions.argmax()

# Print the name of the menu item with the highest predicted sales
print(data["menu_item"].iloc[best_menu_index])
