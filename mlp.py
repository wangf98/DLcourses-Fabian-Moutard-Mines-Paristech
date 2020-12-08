import numpy as np

from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import PrecisionRecallDisplay
import warnings
from sklearn.datasets import fetch_openml


digits = load_digits()
n_samples = len(digits.images)
print("Number_of-examples = ", n_samples)

import matplotlib.pyplot as plt
print("\n Plot of first example")
plt.gray()
plt.matshow(digits.images[0])
print("CLOSE PLOT WINDOW TO CONTINUE")
plt.ioff()
plt.show()

# Flatten the images, to turn data in a (samples, feature) matrix:
data = digits.images.reshape((n_samples, -1))

Xdigits = data
y_digits = digits.target
Xdigits_train, Xdigits_test, y_digits_train, y_digits_test = train_test_split(Xdigits, y_digits, test_size=0.3)

clf = MLPClassifier(hidden_layer_sizes=(32, 64), activation='relu', solver='sgd',
                    alpha=0.00001, batch_size=10, learning_rate='constant', learning_rate_init=0.001,
                    power_t=0.5, max_iter=50, shuffle=True, random_state=11, tol=0.00001,
                    verbose=True, warm_start=False, momentum=0.5, nesterovs_momentum=True,
                    early_stopping=False, validation_fraction=0.1,
                    beta_1=0.9, beta_2=0.999, epsilon=1e-08)
print(clf)

# Train the MLP classifier on training dataset
clf.fit(Xdigits_train, y_digits_train)

# Plot the LEARNING CURVE
plt.title("Evolution of TRAINING ERROR during training")
plt.xlabel("Iterations (epochs)")
plt.ylabel("TRAINING ERROR")
plt.plot(clf.loss_curve_)
plt.show()

# Evaluate acuracy on test data
score = clf.score(Xdigits_test,y_digits_test)
print("Acuracy (on test set) = ", score)
y_true, y_pred = y_digits_test, clf.predict(Xdigits_test)
print(classification_report(y_true, y_pred))

# Display CONFUSION MATRIX on TEST set

print("CONFUSION MATRIX below")
confusion_matrix(y_true, y_pred)


for i in range(10):
    prec, recall, _= precision_recall_curve(y_true, y_pred, pos_label = clf.classes_[i])
    pr_display = PrecisionRecallDisplay(precision = prec, recall = recall).plot()

param_grid = [
    {'hidden_layer_sizes': [(32,), (64,)],
     'learning_rate_init': [0.001, 0.01],
     'alpha': [0.00001, 0.0001]}
]
print(param_grid)

# Cross-validation grid-search (for finding best possible accuracy)
clf = GridSearchCV(MLPClassifier(activation='relu', alpha=1e-07, batch_size=4, beta_1=0.9,
                                 beta_2=0.999, early_stopping=True, epsilon=1e-08,
                                 hidden_layer_sizes=(32, 64), learning_rate='constant',
                                 learning_rate_init=0.005, max_iter=500, momentum=0.8,
                                 nesterovs_momentum=True, power_t=0.5, random_state=11, shuffle=True,
                                 solver='adam', tol=1e-05, validation_fraction=0.3, verbose=False,
                                 warm_start=False),
                   param_grid, cv=3, scoring='accuracy')
# NOTE THAT YOU CAN USE OTHER VALUE FOR cv (# of folds) and OTHER SCORING CRITERIA OTHER THAN 'accuracy'

clf.fit(Xdigits_train, y_digits_train)
print("Best parameters set found on development set:")
print()
print(clf.best_params_)
print()
print("Grid scores on development set:")
print()
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))
print()
print("Detailed classification report:")
print()
print("The model is trained on the full development set.")
print("The scores are computed on the full evaluation set.")
print()
y_true, y_pred = y_digits_test, clf.predict(Xdigits_test)
print(classification_report(y_true, y_pred))
print()

for i in range(10):
    prec, recall, _= precision_recall_curve(y_true, y_pred, pos_label = clf.classes_[i])
    pr_display = PrecisionRecallDisplay(precision = prec, recall = recall).plot()


titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(clf, Xdigits_test, y_digits_test,
                                 display_labels=clf.classes_,
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)

plt.show()

fig, axes = plt.subplots(8, 4)
vmin, vmax = clf.coefs_[0].min(), clf.coefs_[0].max()
for coef, ax in zip(clf.coefs_[0].T, axes.ravel()):
    ax.matshow(coef.reshape(28, 28), cmap=plt.cm.gray, vmin=.5 * vmin,
               vmax=.5 * vmax)
    ax.set_xticks(())
    ax.set_yticks(())

plt.show()