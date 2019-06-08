from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import SVC, LinearSVC
from joblib import dump, load
from sample import getMagnitude
import csv, re, numpy as np

PITCH_CLASSES = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']


def main():
    # test_model('det1')
    # create_neighbors_model(neighbors=1, name='det1')
    # create_SGD_model('sgd')
    # score = test_model(model='sgd')
    # print(score)
    create_neighbors_model(name='model')
    # neigh = load('det1')
    # test = np.array(getMagnitude('Gb5.mp3'))[:, :175].reshape(1, -1)
    # print(neigh.predict(test))


def test_model(model='model'):
    model = load(model)

    X, y = setup_test()
    nsamples, nx, ny = X.shape
    X = X.reshape((nsamples, nx * ny))
    prediction = model.predict(X)
    print(classification_report(y, prediction))
    print(accuracy_score(y, prediction))
    return prediction


def create_SVC_model(name='model'):
    clf = SVC(gamma='auto')
    X, y = generate_dataset()
    nsamples, nx, ny = X.shape
    X = X.reshape((nsamples, nx * ny))
    clf.fit(X, y)
    dump(clf, name, compress=3)


def create_linearSVC_model(name='model'):
    clf = LinearSVC(random_state=0, tol=1e-5)
    X, y = generate_dataset()
    nsamples, nx, ny = X.shape
    X = X.reshape((nsamples, nx * ny))
    clf.fit(X, y)
    dump(clf, name, compress=3)


def create_neighbors_model(neighbors=1, name='model'):
    neigh = KNeighborsClassifier(n_neighbors=neighbors)
    X, y = generate_dataset()
    nsamples, nx, ny = X.shape
    X = X.reshape((nsamples, nx * ny))
    neigh.fit(X, y)
    dump(neigh, name, compress=3)


def create_SGD_model(name):
    sgd = linear_model.SGDClassifier(max_iter=1000)
    X, y = generate_dataset()

    nsamples, nx, ny = X.shape
    X = X.reshape((nsamples, nx * ny))
    sgd.fit(X, y)
    dump(sgd, name, compress=3)


def setup_test():
    f = open("resources/test", "r")
    contents = f.read().splitlines()
    X, y = [], []
    for filename in contents:
        X.append(getMagnitude('resources/test-set/{}'.format(filename))[:, :175])
        y.append(pitch_class_from_file(filename))
    return np.array(X), np.array(y)


def read_from_csv():
    with open('target.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            pass


def create_csv(name, data):
    with open(name, 'w') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(data)

    csvFile.close()


def pitch_class_from_file(pitch):
    pitch = pitch.split('.')[0]
    match = re.match(r"([a-z]+)([0-9]+)", pitch, re.I)
    items = match.groups()
    return items[0]


def generate_dataset():
    f = open("resources/training", "r")
    contents = f.read().splitlines()
    X, y = [], []
    for filename in contents:
        X.append(getMagnitude('resources/training-set/{}'.format(filename)))
        y.append(pitch_class_from_file(filename))
    return np.array(X), np.array(y)


if __name__ == '__main__':
    main()
