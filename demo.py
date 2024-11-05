# demo.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Suppress TensorFlow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

def generate_data(num_samples):
    """
    Generate synthetic 2D data points and label them based on whether
    they are inside a circle centered at (0, 0) with radius 0.5.
    """
    X = np.random.uniform(-1, 1, (num_samples, 2))
    Y = np.zeros((num_samples, 1))

    for i in range(num_samples):
        x, y = X[i]
        # Equation of a circle: x^2 + y^2 = r^2
        if x**2 + y**2 <= 0.5**2:
            Y[i] = 1  # Inside the circle
        else:
            Y[i] = 0  # Outside the circle
    return X, Y

def build_model():
    """
    Build a simple neural network with one hidden layer.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(8, activation='relu', input_shape=(2,)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

def main():
    # Step 1: Generate Data
    num_samples = 5000
    X, Y = generate_data(num_samples)

    # Split the data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    # Step 2: Build the Model
    model = build_model()

    # Step 3: Compile the Model
    model.compile(
        optimizer='sgd',  # Stochastic Gradient Descent
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # Step 4: Train the Model
    history = model.fit(
        X_train, Y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_test, Y_test),
        verbose=1
    )

    # Step 5: Evaluate the Model
    loss, accuracy = model.evaluate(X_test, Y_test, verbose=0)
    print(f'\nTest Accuracy: {accuracy * 100:.2f}%')

    # Step 6: Visualize the Decision Boundary
    # Create a grid to plot decision boundary
    xx, yy = np.mgrid[-1:1:0.01, -1:1:0.01]
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = model.predict(grid).reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    contour = plt.contourf(xx, yy, probs, levels=[0, 0.5, 1], cmap="RdBu", alpha=0.6)
    plt.colorbar(contour)

    # Plot training points
    plt.scatter(X_train[:, 0], X_train[:, 1], c=Y_train[:, 0], cmap="RdBu", edgecolors='k', alpha=0.5)
    plt.title('Decision Boundary and Training Data')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

if __name__ == '__main__':
    main()
