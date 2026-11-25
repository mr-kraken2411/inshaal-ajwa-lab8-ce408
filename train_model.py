"""Train a Random Forest on the Iris dataset and save it to disk.
This is only the application payload for the deployment lab.
The point of Week 8 is deployment and IAM, not machine learning.
"""
import joblib
import sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
# 1. Load the built-in Iris dataset (150 flowers, 4 measurements each)
iris = load_iris()
X, y = iris.data, iris.target
# 2. Hold back 20% of the rows to test with.
#    random_state=42 makes the split identical every run, so your
#    accuracy number is reproducible in a teaching lab.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# 3. Train
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
# 4. Report accuracy on the held-back rows
accuracy = model.score(X_test, y_test)
print("Test accuracy: {:.4f}".format(accuracy))
print("Classes: {}".format(iris.target_names.tolist()))
# 5. Save the trained model to a file.
#    The scikit-learn version is printed because the model must be
#    loaded by the SAME version on the EC2 instance. If this does not
#    match the pin in requirements.txt, retrain in an environment
#    that was built from requirements.txt.
joblib.dump(model, "iris_model.pkl")
print("Saved iris_model.pkl")
print("Trained with scikit-learn {} - this MUST match requirements.txt".format(
    sklearn.__version__))
