"""
COMP-5710 Workshop 8: Forensics
Author: Jacob Murrah
Date: 10/27/2025
"""


from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets, linear_model
import pandas as pd
import numpy as np
import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.utils import to_categorical

from logger import logger


def readData():
    # Security log 1/20: Track dataset ingestion attempt to tie resource access to Dolly's identity realm.
    logger.info(
        "Starting Iris dataset ingestion for Dolly",
        user="Dolly",
        realm="Auburn",
        system="MLPipeline",
        component="DataLoader",
        status="START",
        context="DatasetRead",
        source="sklearn.datasets",
    )

    iris = datasets.load_iris()
    print(type(iris.data), type(iris.target))
    X = iris.data
    Y = iris.target
    df = pd.DataFrame(X, columns=iris.feature_names)
    print(df.head())

    # Security log 2/20: Confirm dataset shape so resource tampering causing poison-sized changes is auditable.
    try:
        logger.info(
            "Iris dataset loaded with %d samples and %d features",
            iris.data.shape[0],
            iris.data.shape[1],
            user="Dolly",
            realm="Auburn",
            system="MLPipeline",
            component="DataLoader",
            status="SUCCESS",
            context="DatasetRead",
            source="sklearn.datasets",
        )
    except Exception:
        logger.exception(
            "Iris dataset shape log failed, resource integrity check incomplete",
            user="Dolly",
            realm="Auburn",
            system="MLPipeline",
            component="DataLoader",
            status="LOG_FAILURE",
            context="DatasetRead",
            source="sklearn.datasets",
        )
    # Security log 3/20: Persist preview of fields to detect unexpected schema shifts from compromised files.
    logger.info(
        "Iris feature preview snapshot=%s",
        df.head(1).to_dict(),
        user="Dolly",
        realm="Auburn",
        system="MLPipeline",
        component="DataLoader",
        status="PROFILED",
        context="DatasetProfiling",
        source="memory",
    )
    # Security log 4/20: Record NaN scan outcome to flag poisoning via corrupted numeric resources.
    logger.info(
        "Iris NaN scan flagged=%s",
        bool(np.isnan(X).any()),
        user="Dolly",
        realm="Auburn",
        system="MLPipeline",
        component="DataLoader",
        status="VALIDATION_ALERT" if np.isnan(X).any() else "VALIDATED",
        context="DatasetValidation",
        source="memory",
    )
    # Security log 5/20: Document duplication count so change issues in source files are observable.
    logger.info(
        "Iris duplicated rows detected=%d",
        int(df.duplicated().sum()),
        user="Dolly",
        realm="Auburn",
        system="MLPipeline",
        component="DataLoader",
        status="ANOMALY" if df.duplicated().any() else "VERIFIED",
        context="DatasetValidation",
        source="memory",
    )

    return df


def makePrediction():
    # Security log 6/20: Log classifier pipeline start with component context for accountability.
    logger.info(
        "Beginning KNN training sequence on Iris dataset",
        user="Dolly",
        realm="Auburn",
        system="MLPipeline",
        component="KNNModel",
        status="START",
        context="ModelTraining",
        source="sklearn.datasets",
    )

    iris = datasets.load_iris()
    knn = KNeighborsClassifier(n_neighbors=6)
    # Security log 7/20: Capture class balance to detect poisoning-induced label skew.
    logger.info(
        "Observed Iris class distribution=%s",
        np.bincount(iris["target"]).tolist(),
        user="Dolly",
        realm="Auburn",
        system="MLPipeline",
        component="KNNModel",
        status="PROFILED",
        context="DataIntegrity",
        source="memory",
    )
    knn.fit(iris["data"], iris["target"])
    # Security log 8/20: Record training accuracy to surface model tricking symptoms.
    try:
        logger.info(
            "KNN training accuracy=%.4f",
            knn.score(iris["data"], iris["target"]),
            user="Dolly",
            realm="Auburn",
            system="MLPipeline",
            component="KNNModel",
            status=(
                "PREDICT_RISK"
                if knn.score(iris["data"], iris["target"]) < 0.9
                else "TRAINED"
            ),
            context="ModelTraining",
            source="cpu",
        )
    except Exception:
        logger.exception(
            "KNN accuracy logging failed, exception masks model tricking detection",
            user="Dolly",
            realm="Auburn",
            system="MLPipeline",
            component="KNNModel",
            status="LOG_FAILURE",
            context="ModelTraining",
            source="cpu",
        )
    X = [
        [5.9, 1.0, 5.1, 1.8],
        [3.4, 2.0, 1.1, 4.8],
    ]
    # Security log 9/20: Log inference request source to track downstream usage.
    logger.info(
        "Executing KNN inference on %d samples",
        len(X),
        user="Dolly",
        realm="Auburn",
        system="MLPipeline",
        component="KNNModel",
        status="INFERENCE_START",
        context="ModelPrediction",
        source="cpu",
    )
    prediction = knn.predict(X)
    print(prediction)
    # Security log 10/20: Record prediction diversity to detect uniform outputs caused by model tricking.
    logger.info(
        "KNN predictions=%s unique_classes=%s",
        prediction.tolist(),
        np.unique(prediction).tolist(),
        user="Dolly",
        realm="Auburn",
        system="MLPipeline",
        component="KNNModel",
        status=(
            "PREDICT_ALERT" if len(np.unique(prediction)) == 1 else "PREDICT_SUCCESS"
        ),
        context="ModelPrediction",
        source="cpu",
    )


def doRegression():
    # Security log 11/20: Note diabetes regression pipeline start to correlate with initiating realm.
    logger.info(
        "Starting linear regression workflow on Diabetes dataset",
        user="Dolly",
        realm="Auburn",
        system="MLPipeline",
        component="LinearRegressionModel",
        status="START",
        context="ModelTraining",
        source="sklearn.datasets",
    )

    diabetes = datasets.load_diabetes()
    # Security log 12/20: Record dataset dimensionality to track resource drift tied to poisoning.
    logger.info(
        "Diabetes dataset samples=%d features=%d",
        diabetes.data.shape[0],
        diabetes.data.shape[1],
        user="Dolly",
        realm="Auburn",
        system="MLPipeline",
        component="LinearRegressionModel",
        status="SUCCESS",
        context="DatasetRead",
        source="sklearn.datasets",
    )
    diabetes_X = diabetes.data[:, np.newaxis, 2]
    diabetes_X_train = diabetes_X[:-20]
    diabetes_X_test = diabetes_X[-20:]
    diabetes_y_train = diabetes.target[:-20]
    diabetes_y_test = diabetes.target[-20:]
    # Security log 13/20: Persist target statistics to detect sudden label shifts from tampering.
    logger.info(
        "Diabetes training target mean=%.4f std=%.4f",
        diabetes_y_train.mean(),
        diabetes_y_train.std(),
        user="Dolly",
        realm="Auburn",
        system="MLPipeline",
        component="LinearRegressionModel",
        status="PROFILED",
        context="DataIntegrity",
        source="memory",
    )
    regr = linear_model.LinearRegression()
    regr.fit(diabetes_X_train, diabetes_y_train)
    # Security log 14/20: Log coefficient magnitude to expose potential poisoning spikes.
    logger.info(
        "Linear regression coefficient=%s",
        regr.coef_.tolist(),
        user="Dolly",
        realm="Auburn",
        system="MLPipeline",
        component="LinearRegressionModel",
        status="COEFF_ALERT" if np.any(np.abs(regr.coef_) > 500) else "TRAINED",
        context="ModelTraining",
        source="cpu",
    )
    diabetes_y_pred = regr.predict(diabetes_X_test)
    # Security log 15/20: Monitor prediction residuals to flag tricking when performance degrades.
    logger.info(
        "Linear regression test MSE=%.4f",
        np.mean((diabetes_y_pred - diabetes_y_test) ** 2),
        user="Dolly",
        realm="Auburn",
        system="MLPipeline",
        component="LinearRegressionModel",
        status=(
            "PERF_REGRESSION"
            if np.mean((diabetes_y_pred - diabetes_y_test) ** 2) > 6000
            else "PERF_OK"
        ),
        context="ModelEvaluation",
        source="cpu",
    )


def doDeepLearning():
    # Security log 16/20: Document CNN workflow kickoff across distributed MNIST source.
    logger.info(
        "Initializing CNN training workflow on MNIST dataset",
        user="Dolly",
        realm="Auburn",
        system="MLPipeline",
        component="CNNModel",
        status="START",
        context="ModelTraining",
        source="mnist",
    )

    train_images = mnist.train_images()
    train_labels = mnist.train_labels()
    test_images = mnist.test_images()
    test_labels = mnist.test_labels()
    # Security log 17/20: Record raw dataset dimensions to baseline resource integrity.
    logger.info(
        "MNIST raw shapes train=%s test=%s",
        tuple(train_images.shape),
        tuple(test_images.shape),
        user="Dolly",
        realm="Auburn",
        system="MLPipeline",
        component="CNNModel",
        status="PROFILED",
        context="DatasetRead",
        source="mnist",
    )

    train_images = (train_images / 255) - 0.5
    test_images = (test_images / 255) - 0.5

    # Security log 18/20: Capture normalization bounds to spot poisoning via unexpected ranges.
    logger.info(
        "MNIST normalized range train=(%.4f, %.4f) test=(%.4f, %.4f)",
        float(train_images.min()),
        float(train_images.max()),
        float(test_images.min()),
        float(test_images.max()),
        user="Dolly",
        realm="Auburn",
        system="MLPipeline",
        component="CNNModel",
        status="NORMALIZED",
        context="DataIntegrity",
        source="memory",
    )

    train_images = np.expand_dims(train_images, axis=3)
    test_images = np.expand_dims(test_images, axis=3)

    # Security log 19/20: Log tensor reshape outcome to trace preprocessing change issues.
    try:
        logger.info(
            "MNIST tensor shapes after expand_dims train=%s test=%s",
            tuple(train_images.shape),
            tuple(test_images.shape),
            user="Dolly",
            realm="Auburn",
            system="MLPipeline",
            component="CNNModel",
            status="PREPROCESSED",
            context="DataPreparation",
            source="cpu",
        )
    except Exception:
        logger.exception(
            "MNIST tensor reshape logging failed, change tracking visibility reduced",
            user="Dolly",
            realm="Auburn",
            system="MLPipeline",
            component="CNNModel",
            status="LOG_FAILURE",
            context="DataPreparation",
            source="cpu",
        )

    num_filters = 8
    filter_size = 3
    pool_size = 2

    model = Sequential(
        [
            Conv2D(num_filters, filter_size, input_shape=(28, 28, 1)),
            MaxPooling2D(pool_size=pool_size),
            Flatten(),
            Dense(10, activation="softmax"),
        ]
    )

    # Security log 20/20: Capture model compile context and optimizer identity provider.
    logger.info(
        "CNN compiled with optimizer=%s loss=%s metrics=%s",
        "adam",
        "categorical_crossentropy",
        ["accuracy"],
        user="Dolly",
        realm="Auburn",
        system="MLPipeline",
        component="CNNModel",
        status="COMPILED",
        context="ModelSetup",
        source="cpu",
    )

    # Compile the model.
    model.compile(
        "adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Train the model.
    model.fit(
        train_images,
        to_categorical(train_labels),
        epochs=3,
        validation_data=(test_images, to_categorical(test_labels)),
    )

    model.save_weights("cnn.h5")

    predictions = model.predict(test_images[:5])

    print(np.argmax(predictions, axis=1))  # [7, 2, 1, 0, 4]

    print(test_labels[:5])  # [7, 2, 1, 0, 4]


if __name__ == "__main__":
    data_frame = readData()
    makePrediction()
    doRegression()
    doDeepLearning()
