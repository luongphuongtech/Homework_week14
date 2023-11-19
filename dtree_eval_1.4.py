import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

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

    # Initialize arrays to store accuracy values for each classifier
    decision_tree_accuracies = []
    decision_stump_accuracies = []
    dt3_accuracies = []

    num_trials = 100
    num_folds = 10

    learning_curve_data = []

    for _ in range(num_trials):
        # Split the data into 10 folds
        folds = np.array_split(np.arange(n), num_folds)

        trial_learning_curve = []

        for fold in folds:
            fold_learning_curve = []

            # Divide the data into training and testing sets for this fold
            train_indices = np.setdiff1d(np.arange(n), fold)
            Xtrain, Xtest = X[train_indices], X[fold]
            ytrain, ytest = y[train_indices], y[fold]

            # Train the decision tree classifier
            for percent in range(10, 101, 10):
                num_train_samples = int(len(train_indices) * percent / 100)
                Xtrain_subset, ytrain_subset = X[train_indices[:num_train_samples]], y[train_indices[:num_train_samples]]

                clf = DecisionTreeClassifier()
                clf.fit(Xtrain_subset, ytrain_subset)

                # Output predictions on the remaining data
                y_pred = clf.predict(Xtest)

                # Compute accuracy for decision tree
                decision_tree_accuracy = accuracy_score(ytest, y_pred)
                fold_learning_curve.append(decision_tree_accuracy)

            trial_learning_curve.append(fold_learning_curve)

        learning_curve_data.append(np.mean(trial_learning_curve, axis=0))

    # Calculate mean and standard deviation for each classifier
    mean_decision_tree_accuracy = np.mean(decision_tree_accuracies)
    stddev_decision_tree_accuracy = np.std(decision_tree_accuracies)

    # Create and return the matrix of statistics
    stats = np.array([
        [mean_decision_tree_accuracy, stddev_decision_tree_accuracy],
        [mean_decision_stump_accuracy, stddev_decision_stump_accuracy],
        [mean_dt3_accuracy, stddev_dt3_accuracy]
    ])

    # Plot learning curves
    plot_learning_curves(learning_curve_data)

    return stats


def plot_learning_curves(data):
    percent_range = range(10, 101, 10)

    mean_values = np.mean(data, axis=0)
    std_dev_values = np.std(data, axis=0)

    plt.errorbar(percent_range, mean_values, yerr=std_dev_values, label='Decision Tree')

    # Add more classifiers and depths as needed
    # Example: Decision Stump, 3-level Decision Tree, and Decision Tree with depth 5
    plt.errorbar(percent_range, mean_values_stump, yerr=std_dev_values_stump, label='Decision Stump')
    plt.errorbar(percent_range, mean_values_dt3, yerr=std_dev_values_dt3, label='3-level Decision Tree')
    plt.errorbar(percent_range, mean_values_dt5, yerr=std_dev_values_dt5, label='Decision Tree (Depth 5)')

    plt.xlabel('Percentage of Training Data')
    plt.ylabel('Mean Accuracy')
    plt.title('Learning Curve for Different Classifiers')
    plt.legend()
    plt.show()


# Do not modify from HERE...
if __name__ == "__main__":
    stats = evaluatePerformance()
    print("Decision Tree Accuracy = ", stats[0, 0], " (", stats[0, 1], ")")
    print("Decision Stump Accuracy = ", stats[1, 0], " (", stats[1, 1], ")")
    print("3-level Decision Tree Accuracy = ", stats[2, 0], " (", stats[2, 1], ")")
# ...to HERE.