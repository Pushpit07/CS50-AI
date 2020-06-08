import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    print("Making Predictions...")
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)
    
    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")
    

def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    print("Loading data...")
    with open(filename) as f:
        reader = csv.reader(f)
        
        data =([],[])
        next(reader)
        for row in reader:
            data[0].append(
                [
                    (int(row[0])),
                    (float(row[1])),
                    (int(row[2])),
                    (float(row[3])),
                    (int(row[4])),
                    (float(row[5])),
                    (float(row[6])),
                    (float(row[7])),
                    (float(row[8])),
                    (float(row[9])),
                    (int(0) if row[10] == "Jan" 
                            else int(1) if row[10] == "Feb" 
                            else int(2) if row[10] == "Mar" 
                            else int(3) if row[10] == "Apr" #Can check here by replacing with April
                            else int(4) if row[10] == "May"
                            else int(5) if row[10] == "June"
                            else int(6) if row[10] == "Jul"
                            else int(7) if row[10] == "Aug"
                            else int(8) if row[10] == "Sep"
                            else int(9) if row[10] == "Oct"
                            else int(10) if row[10]== "Nov"
                            else int(11)
                    ),
                    (int(row[11])),
                    (int(row[12])),
                    (int(row[13])),
                    (int(row[14])),
                    (int(1) if row[15] == "Returning_Visitor" else int(0)),
                    (int(1) if row[16] == "TRUE" else int(0))
                ]
            )
            
            data[1].append(int(1) if row[17] == "TRUE" else int(0))
            
        return (data[0], data[1])

def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    print("Training the model...")
    model = KNeighborsClassifier(n_neighbors=1)
    return model.fit(evidence, labels)


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificty).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    actual_positive = 0
    actual_negative = 0
    correct_pos = 0
    correct_neg = 0

    for label in labels:
        if label == 1:
            actual_positive+=1
        elif label == 0:
            actual_negative+=1

    for actual, predicted in zip(labels, predictions):
        if actual == 1 and actual == predicted:
            correct_pos+=1
        elif actual == 0 and actual == predicted:
            correct_neg+=1
    """
    print("Pos:")
    print(actual_positive)
    print("Neg:")
    print(actual_negative)
    """
    sensitivity = correct_pos/actual_positive
    #print("sensitivity: ")
    #print(sensitivity )

    specificity = correct_neg/actual_negative

    return (sensitivity, specificity)


if __name__ == "__main__":
    main()
