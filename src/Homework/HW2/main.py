import os
import time
import matplotlib.pyplot as plt  # For plotting confusion matrix
import numpy as np

from newVectorizer import vectorize_text
from newVectorizer import train_naive_bayes, evaluate_classifier


def load_documents(path, categories):
    """
    Loads text files from the given directory for specified categories.

    Parameters:
      path (str): Root folder (train or test directory).
      categories (list of str): List of folder names corresponding to the target classes.

    Returns:
      documents (list of str): List of document texts.
      labels (list of str): Corresponding folder names (labels) for each document.
    """
    documents, labels = [], []
    for cat in categories:
        folder = os.path.join(path, cat)
        for filename in os.listdir(folder):
            filepath = os.path.join(folder, filename)
            with open(filepath, 'r', encoding='utf8', errors='ignore') as f:
                documents.append(f.read())
                labels.append(cat)
    return documents, labels


def encode_labels(labels, label_map):
    """
    Converts string labels into numeric labels based on a mapping.

    Parameters:
      labels (list of str): Original string labels.
      label_map (dict): Mapping from label names to numeric codes.

    Returns:
      list: Numeric labels.
    """
    return [label_map[label] for label in labels]


def get_misclassified_examples(documents, y_true, y_pred, n=3):
    """
    Retrieves the first n misclassified examples.

    Parameters:
      documents (list of str): Original documents (test set).
      y_true (list or array): True numeric labels.
      y_pred (list or array): Predicted numeric labels.
      n (int): Number of examples to retrieve.

    Returns:
      List of tuples: (index, document, true_label, predicted_label)
    """
    misclassified_idx = [i for i in range(len(y_true)) if y_true[i] != y_pred[i]]
    chosen_idx = misclassified_idx[:n]
    examples = [(i, documents[i], y_true[i], y_pred[i]) for i in chosen_idx]
    return examples





def main():
    overall_start = time.time()

    # Paths to the train and test folders (adjust these paths accordingly)
    train_path = "20news-bydate-train"
    test_path = "20news-bydate-test"

    # Specify the two categories to use
    categories = ["rec.autos", "rec.motorcycles"]

    # 1) Load the documents and their string labels
    start = time.time()
    train_docs, train_labels = load_documents(train_path, categories)
    test_docs, test_labels = load_documents(test_path, categories)
    print(f"Data loading completed in {time.time() - start:.2f} seconds.")

    # 2) Encode the string labels as numeric (rec.autos -> 0, rec.motorcycles -> 1)
    label_map = {"rec.autos": 0, "rec.motorcycles": 1}
    y_train = encode_labels(train_labels, label_map)
    y_test = encode_labels(test_labels, label_map)

    # 3) Vectorize the text documents
    start = time.time()
    X_train, X_test, vectorizer = vectorize_text(train_docs, test_docs)
    print(f"Vectorization completed in {time.time() - start:.2f} seconds.")

    # 4) Train the Naïve Bayes classifier
    start = time.time()
    clf = train_naive_bayes(X_train, y_train, alpha=0.01)
    print(f"Classifier training completed in {time.time() - start:.2f} seconds.")

    # 5) Evaluate the classifier (using rec.autos as positive class: pos_label=0)
    start = time.time()
    f1, y_pred = evaluate_classifier(clf, X_test, y_test, pos_label=0, plot_cm=True)
    eval_time = time.time() - start
    print(f"Evaluation completed in {eval_time:.2f} seconds.")

    # Display the confusion matrix plot
    plt.show()

    # 6) Identify three misclassified documents
    misclassified_examples = get_misclassified_examples(test_docs, y_test, y_pred, n=3)

    # 7) For each misclassified example, generate and print a prompt for LLM evaluation
    print("\n--- Misclassified Document Analysis ---\n")
    for idx, doc, true_label, nb_pred in misclassified_examples:
        # Convert numeric labels back to string labels using inverse mapping
        inv_label_map = {v: k for k, v in label_map.items()}
        true_label_str = inv_label_map[true_label]
        nb_pred_str = inv_label_map[nb_pred]



    overall_time = time.time() - overall_start
    print(f"\nTotal running time: {overall_time:.2f} seconds.")

    print("\nDone!")


if __name__ == "__main__":
    main()
