import numpy as np

def accuracy_score(y_true, y_pred, threshold=0.5):
    """
    Compute classification accuracy for binary classification.

    Parameters
    ----------
    y_true : numpy.ndarray
        True labels with shape (n_samples, 1) or (n_samples,).
        Expected values are 0 or 1.

    y_pred : numpy.ndarray
        Predicted probabilities from the network, usually the output
        of a sigmoid activation. Shape should match y_true.

    threshold : float, default=0.5
        Probability threshold used to convert predicted probabilities
        into class labels:
            - if prediction >= threshold --> class 1
            - if prediction < threshold  --> class 0

    Returns
    -------
    float
        Classification accuracy in the range [0, 1].

    Explanation
    -----------
    Accuracy measures the fraction of predictions that are correct.

    For binary classification, the network usually outputs probabilities,
    not class labels directly. So we first convert probabilities into
    0 or 1 using a threshold, then compare them to the true labels.
    """

    # Convert inputs to NumPy arrays and flatten them into 1D arrays.
    #
    # This makes it easier to compare values element-by-element,
    # regardless of whether the original shape was:
    # - (n_samples,)
    # - (n_samples, 1)
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    # Convert predicted probabilities into binary class labels.
    #
    # Example with threshold = 0.5:
    # - 0.80 becomes 1
    # - 0.32 becomes 0
    # - 0.50 becomes 1
    y_pred_labels = (y_pred >= threshold).astype(int)

    # Compare true labels and predicted labels.
    # The expression (y_true == y_pred_labels) produces an array
    # of True/False values, and np.mean(...) converts that into
    # the fraction of correct predictions.
    accuracy = np.mean(y_true == y_pred_labels)

    return accuracy


def mean_absolute_error(y_true, y_pred):
    """
    Compute Mean Absolute Error (MAE) for regression.

    Parameters
    ----------
    y_true : numpy.ndarray
        True target values with shape (n_samples, 1) or (n_samples,).

    y_pred : numpy.ndarray
        Predicted target values with shape matching y_true.

    Returns
    -------
    float
        Mean absolute error.

    Explanation
    -----------
    MAE measures the average size of the prediction error.

    It computes:
        mean(|y_true - y_pred|)

    This tells us, on average, how far the predictions are
    from the true target values, without caring whether the
    error was positive or negative.
    """

    # Convert inputs to NumPy arrays and flatten them into 1D arrays.
    # This keeps the computation simple and works for shapes like:
    # - (n_samples,)
    # - (n_samples, 1)
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    # Compute the absolute difference for each sample,
    # then average those differences.
    mae = np.mean(np.abs(y_true - y_pred))

    return mae