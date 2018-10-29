import pickle
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_validate


def save_file(file_full_name, file_content):
    with open(file_full_name, 'wb') as handle:
        pickle.dump(file_content, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_file(file_full_name):
    with open(file_full_name, 'rb') as handle:
        file_content = pickle.load(handle)
    return file_content


def model_test_cross(x, y):
    gnb = GaussianNB()
    cv_results = cross_validate(gnb, x, y, cv=5, return_train_score = True)
    print(cv_results)


def model_test(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    gnb = GaussianNB()
    gnb.fit(x_train, y_train)
    y_pred = gnb.predict(x_test)
    print("===========================")
    print('Accuracy: {:.2f}'.format(gnb.score(x_test, y_test)))
    print("===========================")
    print(classification_report(y_test, y_pred))
    print("===========================")
    print(confusion_matrix(y_test, y_pred))
