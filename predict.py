import pickle
import numpy as np
from keras.models import load_model
from sklearn.metrics import accuracy_score

# Load the trained model
model = load_model('Predict_Model.h5')

# Load the test data from the pkl file
#정규화 하지 않은 파일
with open('testdata1D.pkl', 'rb') as file:
    test_data = pickle.load(file)

# Print the content of test_data
#print(test_data)

# Extract the necessary data from test_data
X_test = test_data[0]
Y_test = test_data[1]

# Save test data with the correct structure
test_data = {'X_test': X_test, 'Y_test': Y_test}

with open('test_data.pkl', 'wb') as file:
    pickle.dump(test_data, file)

# Access the test data from the tuple

X_test = test_data['X_test']
Y_test = test_data['Y_test']

X_test = X_test.reshape(X_test.shape[0], 784).astype('float32') / 255


# Perform predictions
predictions = model.predict(X_test)

# Convert predictions to class labels
predicted_labels = np.argmax(predictions, axis=1)

# Calculate accuracy
accuracy = accuracy_score(Y_test, predicted_labels)

# Print the test values, predicted values, and accuracy
print("Test Values:\n", Y_test)
print("Predicted Values:\n", predicted_labels)
print("Accuracy:\n", accuracy)