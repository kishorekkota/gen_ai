import spacy
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from spacy.util import minibatch
import random

# Load SpaCy's English model
nlp = spacy.blank("en")

# Function to prepare training data
def prepare_training_data(dataframe, text_column, label_column):
    """
    Prepares training data for SpaCy's text classification model.
    Args:
        dataframe (pd.DataFrame): The input data containing text and labels.
        text_column (str): The column containing text data.
        label_column (str): The column containing labels.
    Returns:
        list: A list of tuples in the format (text, {"cats": {"label": value}}).
    """
    training_data = []
    for _, row in dataframe.iterrows():
        text = row[text_column]
        label = row[label_column]
        training_data.append((text, {"cats": {label: 1.0}}))
    return training_data

# Function to train the intent prediction model
def train_intent_model(training_data, n_iter=10):
    """
    Trains a SpaCy text classification model for intent prediction.
    Args:
        training_data (list): The training data in SpaCy's format.
        n_iter (int): Number of training iterations.
    Returns:
        nlp: The trained SpaCy model.
    """
    # Add the text categorizer to the pipeline
    if "textcat" not in nlp.pipe_names:
        textcat = nlp.add_pipe("textcat", last=True)
    else:
        textcat = nlp.get_pipe("textcat")

    # Add labels to the text categorizer
    for _, annotations in training_data:
        for label in annotations["cats"]:
            textcat.add_label(label)

    # Convert training data to SpaCy's format
    train_data = [(nlp.make_doc(text), annotations) for text, annotations in training_data]

    # Train the model
    optimizer = nlp.begin_training()
    for i in range(n_iter):
        random.shuffle(train_data)
        losses = {}
        for batch in minibatch(train_data, size=8):
            texts, annotations = zip(*batch)
            nlp.update(texts, annotations, sgd=optimizer, losses=losses)
        print(f"Iteration {i + 1}, Loss: {losses['textcat']}")

    return nlp

# Function to evaluate the model
def evaluate_model(nlp, test_data):
    """
    Evaluates the trained SpaCy model on test data.
    Args:
        nlp: The trained SpaCy model.
        test_data (list): The test data in SpaCy's format.
    Returns:
        None
    """
    texts, annotations = zip(*test_data)
    true_labels = [max(ann["cats"], key=ann["cats"].get) for ann in annotations]
    predictions = [max(nlp(text).cats, key=nlp(text).cats.get) for text in texts]
    print(classification_report(true_labels, predictions))

# Load historical activity logs
data = pd.read_csv("historical_activity_logs.csv")  # Replace with your CSV file path

# Prepare the data
data["combined_text"] = data["AgentMemo"] + " " + data["AccountActivity"] + " " + data["PreviousCallData"]
training_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
train_data = prepare_training_data(training_data, text_column="combined_text", label_column="Intent")
test_data = prepare_training_data(test_data, text_column="combined_text", label_column="Intent")

# Train the model
trained_model = train_intent_model(train_data, n_iter=10)

# Evaluate the model
evaluate_model(trained_model, test_data)

# Save the model
trained_model.to_disk("intent_prediction_model")

# Example usage
def predict_intent(text, model_path="intent_prediction_model"):
    """
    Predicts the intent of a given text using the trained model.
    Args:
        text (str): The input text.
        model_path (str): Path to the trained model.
    Returns:
        str: The predicted intent.
    """
    nlp = spacy.load(model_path)
    doc = nlp(text)
    return max(doc.cats, key=doc.cats.get)

# Example prediction
example_text = "Customer requested a PIN reset and reported a suspicious transaction."
predicted_intent = predict_intent(example_text)
print(f"Predicted Intent: {predicted_intent}")