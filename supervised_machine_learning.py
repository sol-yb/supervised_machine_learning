# Import necessary libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Step 1: Load the Iris dataset
iris = load_iris()
X = iris.data  # Features (sepal length, sepal width, petal length, petal width)
y = iris.target  # Target labels (species of iris)

# Step 2: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train a k-Nearest Neighbors (k-NN) classifier
model = KNeighborsClassifier(n_neighbors=3)  # Use 3 nearest neighbors
model.fit(X_train, y_train)

# Step 4: Make predictions on the test set
y_pred = model.predict(X_test)

# Step 5: Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Step 6: Predict the species of a new flower
new_flower = [[5.1, 3.5, 1.4, 0.2]]  # Example features (sepal length, sepal width, petal length, petal width)
predicted_species = model.predict(new_flower)
print(f"Predicted Species: {iris.target_names[predicted_species][0]}")