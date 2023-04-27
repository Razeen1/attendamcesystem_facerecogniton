import os
import numpy as np
import cv2
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Function to extract faces from an image
def extract_faces(image):

    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.3, minNeighbors=5)
    return faces

# Function to train and evaluate SVM model
def train_and_evaluate_SVM(faces, labels):
    X_train, X_test, y_train, y_test = train_test_split(faces, labels, test_size=0.2, stratify=labels, random_state=100)
    # Create an array of degrees to use for the polynomial kernel
    degrees = np.arange(1, 5)

    # Create an empty list to store the accuracy scores
    accuracies = []

    # Train an SVM model with each degree and store the accuracy score
    for degree in degrees:
        svm = SVC(kernel='poly', degree=degree)
        svm.fit(X_train, y_train)
        y_pred = svm.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)

    # Plot the accuracy scores versus the degrees
    plt.plot(degrees, accuracies)
    plt.xlabel('Degree')
    plt.ylabel('Accuracy')
    plt.title('SVM Accuracy vs. Degree')
    plt.show()




    #svm = SVC(kernel='poly', degree=10)
    #svm.fit(X_train, y_train)
    #y_pred = svm.predict(X_test)
    #accuracy = accuracy_score(y_test, y_pred)
    #return accuracy

# Function to train and evaluate KNN model
def train_and_evaluate_KNN(faces, labels):
    X_train, X_test, y_train, y_test = train_test_split(faces, labels, test_size=0.2, stratify=labels, random_state=100)
    #knn = KNeighborsClassifier(n_neighbors=100, metric='minkowski')
    #russellrao', 'manhattan', 'precomputed', 'minkowski', 'nan_euclidean', 'yule', 'euclidean', 'haversine', 'wminkowski', 'hamming', 'sqeuclidean', 'jaccard', 'dice', 'matching', 'cosine', 'sokalsneath', 'kulsinski', 'mahalanobis', 'canberra', 'rogerstanimoto', 'l1', 'cityblock', 'chebyshev', 'seuclidean', 'correlation', 'l2', 'p', 'sokalmichener', 'braycurtis', 'pyfunc', 'infinity
    #knn.fit(X_train, y_train)
    #y_pred = knn.predict(X_test)
    #accuracy = accuracy_score(y_test, y_pred)
    #return accuracy

    neighbors = np.arange(1, 10)
    accuracy = np.empty(len(neighbors))

    for i, k in enumerate(neighbors):
        knn = KNeighborsClassifier(n_neighbors=k, algorithm='kd_tree', p=2)
        knn.fit(X_train, y_train)
        accuracy[i] = knn.score(X_test, y_test)

    plt.title('k-NN accuracy by number of Neighbors')
    plt.plot(neighbors, accuracy)
    plt.xlabel('Number of Neighbors')
    plt.ylabel('Accuracy')
    plt.show()




# Main program
faces = []
labels = []
userlist = os.listdir('static/faces')
for user in userlist:
    for imgname in os.listdir(f'static/faces/{user}'):
        img = cv2.imread(f'static/faces/{user}/{imgname}')
        resized_face = cv2.resize(img, (50, 50))
        faces.append(resized_face.ravel())
        labels.append(user)
faces = np.array(faces)
labels = np.array(labels)

svm_accuracy = train_and_evaluate_SVM(faces, labels)
knn_accuracy = train_and_evaluate_KNN(faces, labels)

print("Accuracy of SVM:", svm_accuracy)
print("Accuracy of KNN:", knn_accuracy)

