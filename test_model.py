
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

from train import train_model


# Test 1: Check whether the model trains successfully
def test_model_trains_successfully():

    model, X_test, y_test, accuracy = train_model()

    assert model is not None
    assert accuracy >= 0.80


# Test 2: Check prediction length
def test_predictions_have_correct_length():

    model, X_test, y_test, accuracy = train_model()

    predictions = model.predict(X_test)

    assert len(predictions) == len(y_test)


# Test 3: Check whether predictions contain valid classes
def test_predictions_are_valid_classes():

    model, X_test, y_test, accuracy = train_model()

    predictions = model.predict(X_test)

    valid_classes = set(load_iris().target)

    assert set(predictions).issubset(valid_classes)


# Test 4: Check whether accuracy is valid
def test_accuracy_is_valid():

    model, X_test, y_test, accuracy = train_model()

    predictions = model.predict(X_test)

    actual_accuracy = accuracy_score(y_test, predictions)

    assert 0.0 <= actual_accuracy <= 1.0
    assert actual_accuracy >= 0.80
