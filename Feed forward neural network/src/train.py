import numpy as np

from src.metrics import accuracy_score, mean_absolute_error
from src.utils import RANDOM_SEED

class Trainer:
    """
    Handles neural network training, validation, and test evaluation.

    This class is responsible for:
    - running the epoch loop
    - creating mini-batches
    - computing training and validation loss
    - updating weights using the chosen optimizer
    - tracking metric history
    - optionally applying early stopping
    - evaluating final test performance

    It is designed to work for both:
    - binary classification
    - regression
    """

    def __init__(
        self,
        network,
        loss_fn,
        optimizer,
        task_type,
        early_stopping=False,
        patience=10,
        min_delta=0.0
    ):
        """
        Initialize the trainer.

        Parameters
        ----------
        network : NeuralNetwork
            The neural network model to train.
        loss_fn : object
            Loss function object with:
            - forward(y_true, y_pred)
            - backward(y_true, y_pred)
        optimizer : object
            Optimizer object with an update(layer) method.
        task_type : str
            Either "classification" or "regression".
        early_stopping : bool, default=False
            Whether to enable early stopping based on validation loss.
        patience : int, default=10
            Number of consecutive non-improving epochs allowed
            before stopping.
        min_delta : float, default=0.0
            Minimum validation-loss improvement required to count
            as a real improvement.
        """
        self.network = network
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.task_type = task_type.lower()

        # Make sure task_type is one of the two supported options
        if self.task_type not in ["classification", "regression"]:
            raise ValueError("task_type must be either 'classification' or 'regression'")

        self.early_stopping = early_stopping
        self.patience = patience
        self.min_delta = min_delta

        # Dedicated random generator used for shuffling training data
        # each epoch in a reproducible way
        self.random = np.random.RandomState(RANDOM_SEED)

        # History dictionary used to store training progress over epochs
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_metric": []
        }

    def _shuffle_data(self, X, y):
        """
        Shuffle the dataset at the start of each epoch.

        Parameters
        ----------
        X : numpy.ndarray
            Input features.
        y : numpy.ndarray
            Targets.

        Returns
        -------
        X_shuffled, y_shuffled : numpy.ndarray
            Shuffled versions of X and y, using the same permutation.
        """
        indices = self.random.permutation(len(X))
        return X[indices], y[indices]

    def _create_batches(self, X, y, batch_size):
        """
        Split the dataset into mini-batches.

        Parameters
        ----------
        X : numpy.ndarray
            Input features.
        y : numpy.ndarray
            Targets.
        batch_size : int
            Number of samples per mini-batch.

        Yields
        ------
        X_batch, y_batch : numpy.ndarray
            One mini-batch at a time.
        """
        for start_idx in range(0, len(X), batch_size):
            end_idx = start_idx + batch_size
            yield X[start_idx:end_idx], y[start_idx:end_idx]

    def _compute_metric(self, y_true, y_pred):
        """
        Compute the correct evaluation metric for the current task.

        For classification:
            accuracy
        For regression:
            mean absolute error (MAE)

        Parameters
        ----------
        y_true : numpy.ndarray
            True targets.
        y_pred : numpy.ndarray
            Predicted outputs.

        Returns
        -------
        float
            Metric value.
        """
        if self.task_type == "classification":
            return accuracy_score(y_true, y_pred)
        else:
            return mean_absolute_error(y_true, y_pred)

    def _get_model_state(self):
        """
        Save a copy of the current trainable weights.

        Returns
        -------
        list
            A list of copied weight matrices, one per trainable layer.

        Why this is needed
        ------------------
        During early stopping, we want to restore the best validation
        checkpoint later. That means we need to save the weights whenever
        validation loss meaningfully improves.
        """
        state = []
        for layer in self.network.get_trainable_layers():
            state.append(layer.weights.copy())
        return state

    def _set_model_state(self, state):
        """
        Restore previously saved trainable weights.

        Parameters
        ----------
        state : list
            List of saved weight matrices.
        """
        for layer, saved_weights in zip(self.network.get_trainable_layers(), state):
            layer.weights = saved_weights.copy()

    def fit(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32, verbose=True):
        """
        Train the network.

        Parameters
        ----------
        X_train, y_train : numpy.ndarray
            Training data and labels.
        X_val, y_val : numpy.ndarray
            Validation data and labels.
        epochs : int, default=100
            Maximum number of training epochs.
        batch_size : int, default=32
            Number of samples per mini-batch.
        verbose : bool, default=True
            Whether to print progress.

        Returns
        -------
        dict
            Training history, including:
            - train_loss
            - val_loss
            - val_metric
            - epochs_ran
            - stopped_early
            - best_epoch
            - best_val_loss
        """
        # Reset history at the start of every fit() call
        # so previous runs do not contaminate the new one
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_metric": []
        }

        # Variables used for early stopping
        best_val_loss = float("inf")
        best_model_state = None
        best_epoch = None
        epochs_without_improvement = 0

        # Main training loop
        for epoch in range(epochs):
            # Shuffle training data at the start of each epoch
            X_train_shuffled, y_train_shuffled = self._shuffle_data(X_train, y_train)

            # Track batch losses so we can average them into one epoch loss
            batch_losses = []

            # Mini-batch training
            for X_batch, y_batch in self._create_batches(X_train_shuffled, y_train_shuffled, batch_size):
                # Forward pass through the network
                y_pred = self.network.forward(X_batch)

                # Compute loss for this batch
                loss = self.loss_fn.forward(y_batch, y_pred)
                batch_losses.append(loss)

                # Compute gradient of the loss with respect to predictions
                grad_loss = self.loss_fn.backward(y_batch, y_pred)

                # Backpropagate through the whole network
                self.network.backward(grad_loss)

                # Update all trainable layers using the optimizer
                for layer in self.network.get_trainable_layers():
                    self.optimizer.update(layer)

            # Average batch losses to get one training loss for the epoch
            train_loss = np.mean(batch_losses)

            # Validation pass after the epoch finishes
            y_val_pred = self.network.forward(X_val)
            val_loss = self.loss_fn.forward(y_val, y_val_pred)
            val_metric = self._compute_metric(y_val, y_val_pred)

            # Save epoch results in history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_metric"].append(val_metric)

            # Optional console output
            if verbose:
                metric_name = "Accuracy" if self.task_type == "classification" else "MAE"
                print(
                    f"Epoch {epoch + 1:03d}/{epochs} | "
                    f"Train Loss: {train_loss:.6f} | "
                    f"Val Loss: {val_loss:.6f} | "
                    f"Val {metric_name}: {val_metric:.6f}"
                )

            # Early stopping logic
            if self.early_stopping:
                # A new best model is saved only if validation loss
                # improves by more than min_delta
                if val_loss < (best_val_loss - self.min_delta):
                    best_val_loss = float(val_loss)
                    best_epoch = epoch + 1
                    best_model_state = self._get_model_state()
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                if verbose:
                    print(
                        f"    [ES] best_val_loss={best_val_loss:.6f} | "
                        f"counter={epochs_without_improvement}/{self.patience}"
                    )

                # Stop if validation loss has not improved enough
                # for 'patience' consecutive epochs
                if epochs_without_improvement >= self.patience:
                    if verbose:
                        print(
                            f"\nEarly stopping triggered at epoch {epoch + 1}. "
                            f"No meaningful validation-loss improvement for {self.patience} consecutive epochs."
                        )
                    break

        # After training ends, restore the best validation checkpoint
        # so final test evaluation uses the best model, not just the last epoch
        if self.early_stopping and best_model_state is not None:
            self._set_model_state(best_model_state)

        # Record how many epochs actually ran
        self.history["epochs_ran"] = len(self.history["train_loss"])

        # True if training ended before reaching the requested max epochs
        self.history["stopped_early"] = self.history["epochs_ran"] < epochs

        # Save best-epoch information.
        #
        # If early stopping was enabled, use the checkpoint-tracking values
        # directly so the reported best epoch/loss match the restored model.
        #
        # If early stopping was not enabled, simply report the raw minimum
        # validation loss observed in history.
        if self.early_stopping and best_epoch is not None:
            self.history["best_epoch"] = best_epoch
            self.history["best_val_loss"] = best_val_loss
        else:
            self.history["best_epoch"] = int(np.argmin(self.history["val_loss"])) + 1
            self.history["best_val_loss"] = float(min(self.history["val_loss"]))

        return self.history

    def evaluate(self, X_test, y_test):
        """
        Evaluate the trained model on the test set.

        Parameters
        ----------
        X_test, y_test : numpy.ndarray
            Test data and true targets.

        Returns
        -------
        dict
            Dictionary containing:
            - test_loss
            - test_metric

        Notes
        -----
        The metric depends on the task:
        - classification -> accuracy
        - regression -> MAE
        """
        # Forward pass on the test set
        y_test_pred = self.network.forward(X_test)

        # Compute test loss and task-specific metric
        test_loss = self.loss_fn.forward(y_test, y_test_pred)
        test_metric = self._compute_metric(y_test, y_test_pred)

        print("\n--- Test Set Evaluation ---")
        print(f"Test Loss: {test_loss:.6f}")

        if self.task_type == "classification":
            print(f"Test Accuracy: {test_metric:.6f}")
        else:
            print(f"Test MAE: {test_metric:.6f}")

        return {
            "test_loss": test_loss,
            "test_metric": test_metric
        }

    def predict(self, X):
        """
        Run inference using the trained network.

        Parameters
        ----------
        X : numpy.ndarray
            Input data.

        Returns
        -------
        numpy.ndarray
            Network predictions.
        """
        return self.network.forward(X)