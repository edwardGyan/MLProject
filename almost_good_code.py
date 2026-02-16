import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


def load_gaussian_2d(path):
    df = pd.read_csv(path)

    # Extract class-specific coordinates
    class0 = df.iloc[:, 0:2].to_numpy()
    class1 = df.iloc[:, 2:4].to_numpy()

    # Labels
    y0 = np.zeros(len(class0))
    y1 = np.ones(len(class1))

    # Combine
    X = np.vstack((class0, class1))
    y = np.concatenate((y0, y1))

    return X, y

def build_model(input_dim, learning_rate=0.001):
    model = Sequential([
        Dense(16, activation='relu', input_dim=input_dim),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model

def run_experiment(X, y, runs=10):

    all_metrics = []

    for run in range(runs):

        # Split
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.4
        )

        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5
        )

        # Standardization
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

        # Model
        model = build_model(input_dim=X.shape[1])

        early_stop = EarlyStopping(
        monitor='val_loss',
        patience=20,
        min_delta=0.001,          # prevents endless micro-training
        restore_best_weights=True
)


        # Train
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            verbose=1,
            callbacks=[early_stop]
        )

        # Predictions
        y_pred = (model.predict(X_test) > 0.5).astype(int)

        # Metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "conf_matrix": confusion_matrix(y_test, y_pred)
        }

        all_metrics.append(metrics)
        




    return all_metrics
    

# results = run_experiment(X, y)
# summarize_results(results)

def summarize_results(results):

    accuracy = [r["accuracy"] for r in results]
    precision = [r["precision"] for r in results]
    recall = [r["recall"] for r in results]
    f1 = [r["f1"] for r in results]

    print("Accuracy Mean:", np.mean(accuracy))
    print("Accuracy Std:", np.std(accuracy))

    print("Precision Mean:", np.mean(precision))
    print("Recall Mean:", np.mean(recall))
    print("F1 Mean:", np.mean(f1))
    # Define the content you want to write
    # content_to_write = """Gaussian Wide Successful"""

    # # Specify the file name (relative to the current directory)
    # file_path = "log.txt"

    # # Open the file in write mode using a context manager
    # try:
    #     with open(file_path, 'w', encoding='utf-8') as file:
    #         file.write(content_to_write)
    #     print(f"File '{file_path}' Progress logged.")
    # except IOError as e:
    #     print(f"An error occurred: {e}")


# Source - https://stackoverflow.com/a/60404117
# Posted by Bálint Cséfalvay
# Retrieved 2026-02-16, License - CC BY-SA 4.0

# with open("log.txt", "r") as file:
#     for line in file:
#         if line[0] == "Gaussian Wide Successful":
#             X, y = load_gaussian_2d("Gaussian 2D Narrow.csv")
#         elif line[0] == "Gaussian Wide and Narrow":
#             X, y = load_gaussian_2d("Gaussian 2D Narrow.csv")
#         elif line[0] == "Gaussian Medium Successful":
#         X, y = load_gaussian_2d("Gaussian 2D Medium.csv")
#     else:
#         print("No Gaussian Data Found")

def plot_training_curves(history, title):

    # Loss
    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f"{title} Loss Curve")
    plt.legend()
    plt.show()

    # Accuracy
    plt.figure()
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title(f"{title} Accuracy Curve")
    plt.legend()
    plt.show()

def plot_decision_boundary(model, scaler, X, y, title):

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_scaled = scaler.transform(grid)

    preds = (model.predict(grid_scaled) > 0.5).astype(int)
    preds = preds.reshape(xx.shape)

    plt.figure()
    plt.contourf(xx, yy, preds, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.title(f"{title} Decision Boundary")
    plt.show()


def plot_probability_surface(model, scaler, X, y, title):

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_scaled = scaler.transform(grid)

    probs = model.predict(grid_scaled)
    probs = probs.reshape(xx.shape)

    plt.figure()
    plt.contourf(xx, yy, probs, levels=50)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
    plt.title(f"{title} Probability Surface")
    plt.colorbar()
    plt.show()


def plot_confusion_matrix(cm, title):

    plt.figure()
    plt.imshow(cm)
    plt.title(f"{title} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.colorbar()
    plt.show()


def visualize_dataset(path):

    X, y = load_gaussian_2d(path)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    model = build_model(input_dim=2)

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=20,
        min_delta=0.001,
        restore_best_weights=True
    )

    history = model.fit(
        X_train_scaled, y_train,
        validation_data=(X_val_scaled, y_val),
        epochs=500,
        verbose=1,
        callbacks=[early_stop]
    )

    plot_training_curves(history, path)

    y_pred = (model.predict(X_test_scaled) > 0.5).astype(int)

    cm = confusion_matrix(y_test, y_pred)

    plot_decision_boundary(model, scaler, X, y, path)

    plot_probability_surface(model, scaler, X, y, path)

    plot_confusion_matrix(cm, path)


if __name__ == "__main__":

    path = "Gaussian 2D Wide.csv"

    X, y = load_gaussian_2d(path)

    results = run_experiment(X, y)

    summarize_results(results)

    visualize_dataset(path)



# results = run_experiment(X, y)

# summarize_results(results)
