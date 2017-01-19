import glob
import pickle
import re

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.datasets import fetch_olivetti_faces
from sklearn.neural_network import MLPClassifier


class NeuralNetwork:
    def __init__(self):
        self.face = 0

    def base_train(self):
        data = self.load_dataset()
        clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(78,), random_state=1, max_iter=800)
        X_train = data[0]
        X_train_1 = []
        for x in X_train:
            X_train_1_v = []
            for y in x:
                X_train_1_v.append(float(y) / 255)
            X_train_1.append(X_train_1_v)

        y_train = data[1]
        model = clf.fit(X_train_1, y_train)
        pickle.dump(model, open('neural_model.sav', 'wb'))
        print model.loss_
        print model.score(X_train_1, y_train)

    def load_dataset(self):
        dictionary = []
        image_list = []
        face_info = []

        prog = re.compile(r'(resized_images/nonface)[0-9]+(.jpg)')

        for image in glob.glob('resized_images/*.jpg'):
            im = Image.open(image).convert('L')
            image_list.append(list(im.getdata()))

            if prog.match(image):
                face_info.append(0)
            else:
                face_info.append(1)

        dictionary.append(image_list)
        dictionary.append(face_info)
        return dictionary

    def generate_dataset(self):
        self.generate_fetch_olivetti_faces()
        self.generate_nonface()
        # self.generate_CIFAR10()
        # self.gerate_BioID()

    def generate_fetch_olivetti_faces(self):
        data = fetch_olivetti_faces()
        dataset = data.data
        reshape_dataset = np.asarray([np.reshape(ds, (64, 64)) for ds in dataset])
        self.resize_images('face', reshape_dataset, 200)

    def generate_nonface(self):
        dataset = []

        for image in glob.glob('nietwarze/*.jpg'):
            im = Image.open(image).convert('L')
            dataset.append(list(im.getdata()))

        reshape_dataset = np.asarray([np.reshape(ds, (64, 64)) for ds in dataset])
        self.resize_images('nonface', reshape_dataset, 400)

    def generate_CIFAR10(self):
        dataload = pickle.load(open('data_batch_1', 'rb'))
        dataset = dataload['data']

        reshape_dataset = np.asarray([np.reshape(ds, (32, 32)) for ds in dataset[:, :1024]])
        self.resize_images('nonface', reshape_dataset, 2000)

    def gerate_BioID(self):
        dataset = []

        for image in glob.glob('twarze/*.pgm'):
            im = Image.open(image).convert('L')
            dataset.append(list(im.getdata()))

        reshape_dataset = np.asarray([np.reshape(ds, (286, 384)) for ds in dataset[:500]])
        self.resize_images('face', reshape_dataset, 500)

    def resize_images(self, name, dataset, range):
        self.face
        for item in dataset[:range]:
            self.face+=1
            plt.imsave('img.jpg', item, cmap='gray')
            img = Image.open('img.jpg')
            img2 = img.resize((64, 64), Image.ANTIALIAS)
            img2.save('resized_images/' + name + str(self.face) + '.jpg')
