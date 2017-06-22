import numpy as np
import scipy.io
import pandas
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn import svm

"Load the .mat file"
AD = scipy.io.loadmat('Mean_ROI_Features.mat')["AD_Feature"] # 198 subjects
MCI = scipy.io.loadmat('Mean_ROI_Features.mat')["MCI_Feature"] # 374 subjects
NC = scipy.io.loadmat('Mean_ROI_Features.mat')["NC_Feature"] # 229 subjects

AllSub_2C = np.concatenate((AD, NC), axis=1) # 93 ROIS * 427 subjects
AllSub_3C = np.concatenate((AllSub_2C, MCI), axis=1) # 93 ROIS * 801 subjects

"Make labels for two classes AD(1), NC(-1)"
Labels_2C = np.ones(427)
Labels_2C[198:] = -1

"Make labels for three classes AD(1), MCI(0) and NC(-1)"
Labels_3C = np.ones(801)
Labels_3C[198:427] = -1
Labels_3C[427:] = 0


"Split the data for cross validation"
def CV(data, labels):
    X_train, X_test, y_train, y_test = train_test_split(data.T, labels, test_size=0.1, random_state=5)
    return X_train, X_test, y_train, y_test


"SVM"
X_train, X_test, y_train, y_test = CV(data=AllSub_2C, labels=Labels_2C)

# clf = svm.SVC(decision_function_shape='ovr')
# clf.fit(X_train, y_train)
# svmscores = clf.score(X_test, y_test)

# svmscores = cross_val_score(clf, X_test, y_test, cv=5)

"For plot"

h = .02  # step size in the mesh

# we create an instance of SVM and fit out data. We do not scale our
# data since we want to plot the support vectors
C = 1.0  # SVM regularization parameter
svc = svm.SVC(kernel='linear', C=C).fit(X_train, y_train)
rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X_train, y_train)
poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X_train, y_train)
lin_svc = svm.LinearSVC(C=C).fit(X_train, y_train)

# create a mesh to plot in
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

aa = np.c_[xx.ravel(), yy.ravel()]
print(aa.shape)

# title for the plots
titles = ['SVC with linear kernel',
          'LinearSVC (linear kernel)',
          'SVC with RBF kernel',
          'SVC with polynomial (degree 3) kernel']


for i, clf in enumerate((svc, lin_svc, rbf_svc, poly_svc)):
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    plt.subplot(2, 2, i + 1)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = clf.predict(X_test)

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    # Z = Z.reshape(X_test.shape)

    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

    # Plot also the training points
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.coolwarm)
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title(titles[i])

plt.show()