from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np

def evaluatePerformance():
    '''
    Evaluate the performance of decision trees,
    averaged over 1,000 trials of 10-fold cross-validation
    
    Return:
      a matrix giving the performance that will contain the following entries:
      stats[0,0] = mean accuracy of decision tree
      stats[0,1] = std deviation of decision tree accuracy
      stats[1,0] = mean accuracy of decision stump
      stats[1,1] = std deviation of decision stump
      stats[2,0] = mean accuracy of 3-level decision tree
      stats[2,1] = std deviation of 3-level decision tree
      
    ** Note that your implementation must follow this API**
    '''
    
    # Load Data
    filename = 'SPECTF.dat'
    data = np.loadtxt(filename, delimiter=',')
    X = data[:, 1:]
    y = np.array([data[:, 0]]).T
    n, d = X.shape

    # shuffle the data
    idx = np.arange(n)
    np.random.seed(13)
    np.random.shuffle(idx)
    X = X[idx]
    y = y[idx]
    
    # Initialize arrays to store accuracy values for each classifier
    decision_tree_accuracies = []
    decision_stump_accuracies = []
    dt3_accuracies = []

    num_trials = 1000
    num_folds = 10

    for _ in range(num_trials):
        # Split the data into 10 folds
        folds = np.array_split(np.arange(n), num_folds)

        for fold in folds:
            # Divide the data into training and testing sets for this fold
            train_indices = np.setdiff1d(np.arange(n), fold)
            Xtrain, Xtest = X[train_indices], X[fold]
            ytrain, ytest = y[train_indices], y[fold]

            # Train the decision tree classifier
            clf = DecisionTreeClassifier()
            clf.fit(Xtrain, ytrain)

            # Output predictions on the remaining data
            y_pred = clf.predict(Xtest)

            # Compute accuracy for decision tree
            decision_tree_accuracy = accuracy_score(ytest, y_pred)
            decision_tree_accuracies.append(decision_tree_accuracy)

            # Compute accuracy for decision stump (1-level decision tree)
            clf_stump = DecisionTreeClassifier(max_depth=1)
            clf_stump.fit(Xtrain, ytrain)
            y_pred_stump = clf_stump.predict(Xtest)
            decision_stump_accuracy = accuracy_score(ytest, y_pred_stump)
            decision_stump_accuracies.append(decision_stump_accuracy)

            # Compute accuracy for 3-level decision tree
            clf_dt3 = DecisionTreeClassifier(max_depth=3)
            clf_dt3.fit(Xtrain, ytrain)
            y_pred_dt3 = clf_dt3.predict(Xtest)
            dt3_accuracy = accuracy_score(ytest, y_pred_dt3)
            dt3_accuracies.append(dt3_accuracy)

    # Calculate mean and standard deviation for each classifier
    mean_decision_tree_accuracy = np.mean(decision_tree_accuracies)
    stddev_decision_tree_accuracy = np.std(decision_tree_accuracies)
    
    mean_decision_stump_accuracy = np.mean(decision_stump_accuracies)
    stddev_decision_stump_accuracy = np.std(decision_stump_accuracies)
    
    mean_dt3_accuracy = np.mean(dt3_accuracies)
    stddev_dt3_accuracy = np.std(dt3_accuracies)

    # Create and return the matrix of statistics
    stats = np.array([
        [mean_decision_tree_accuracy, stddev_decision_tree_accuracy],
        [mean_decision_stump_accuracy, stddev_decision_stump_accuracy],
        [mean_dt3_accuracy, stddev_dt3_accuracy]
    ])

    return stats


# Do not modify from HERE...
if __name__ == "__main__":
    stats = evaluatePerformance()
    print("Decision Tree Accuracy = ", stats[0, 0], " (", stats[0, 1], ")")
    print("Decision Stump Accuracy = ", stats[1, 0], " (", stats[1, 1], ")")
    print("3-level Decision Tree Accuracy = ", stats[2, 0], " (", stats[2, 1], ")")
# ...to HERE.
