from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


def train_naive_bayes(X_train, y_train, alpha=0.01):
    """
    Trains a Multinomial Naïve Bayes classifier.

    Parameters:
      X_train (sparse matrix): Token count matrix for training data.
      y_train (array-like): Numeric labels for training data.
      alpha (float): Smoothing parameter.

    Returns:
      clf (MultinomialNB): Trained Naïve Bayes classifier.
    """
    clf = MultinomialNB(alpha=alpha)
    clf.fit(X_train, y_train)
    return clf


def evaluate_classifier(clf, X_test, y_test, pos_label=0, plot_cm=False):
    """
    Evaluates the classifier using F1-score (with rec.autos as positive),
    prints classification report, and optionally plots a confusion matrix.

    Parameters:
      clf (MultinomialNB): Trained Naïve Bayes classifier.
      X_test (sparse matrix): Token count matrix for test data.
      y_test (array-like): Numeric labels for test data.
      pos_label (int): The label considered as the positive class (default 0 for rec.autos).
      plot_cm (bool): Whether to display a confusion matrix plot.

    Returns:
      f1 (float): F1-score with rec.autos as positive label.
      y_pred (array): Predicted labels for the test set.
    """
    y_pred = clf.predict(X_test)

    # Calculate F1 score
    f1 = f1_score(y_test, y_pred, pos_label=pos_label)

    # Print classification report
    print("Classification Report (assuming rec.autos is positive):")
    print(classification_report(y_test, y_pred, target_names=["rec.autos", "rec.motorcycles"]))

    # Print F1 score separately
    print(f"F1 Score (pos_label={pos_label}): {f1:.4f}")

    # Optionally plot confusion matrix
    if plot_cm:
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=["rec.autos", "rec.motorcycles"])
        disp.plot(cmap="Blues")  # or any other matplotlib colormap
        disp.ax_.set_title("Confusion Matrix")

    return f1, y_pred
