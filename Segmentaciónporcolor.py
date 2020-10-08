import cv2
import numpy as np
import sys
import os

import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture as GMM
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from time import time
import math


def recreate_image(centers, labels, rows, cols):
    d = centers.shape[1]
    image_clusters = np.zeros((rows, cols, d))
    label_idx = 0
    for i in range(rows):
        for j in range(cols):
            image_clusters[i][j] = centers[labels[label_idx]]
            label_idx += 1
    return image_clusters


def color_segmentation(path_file, method):
    image = cv2.imread(path_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Convert to floats instead of the default 8 bits integer coding. Dividing by
    # 255 is important so that plt.imshow behaves works well on float data (need to
    # be in the range [0-1])
    image = np.array(image, dtype=np.float64) / 255

    # Load Image and transform to a 2D numpy array.
    rows, cols, ch = image.shape
    assert ch == 3
    image_array = np.reshape(image, (rows * cols, ch))
    intraCluster = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    # Display all results, alongside original image
    plt.figure(1)
    plt.clf()
    plt.axis('off')
    plt.title('Original image')
    plt.imshow(image)

    for j in range(10):
        n_colors = j + 1
        print(n_colors)
        print("Fitting model on a small sub-sample of the data")
        t0 = time()
        image_array_sample = shuffle(image_array, random_state=0)[:10000]
        if method == 'gmm':
            model = GMM(n_components=n_colors).fit(image_array_sample)
        else:
            model = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)
        print("done in %0.3fs." % (time() - t0))

        # Get labels for all points
        print("Predicting color indices on the full image (GMM)")
        t0 = time()
        if method == 'gmm':
            labels = model.predict(image_array)
            centers = model.means_
            for i in range(len(labels)):
                if labels[i] == 0:
                    intraCluster[j] += (math.sqrt(
                        np.power(image_array[i][0] - centers[0][0], 2) + np.power(image_array[i][1] - centers[0][1],
                                                                                  2) + np.power(
                            image_array[i][2] - centers[0][2], 2)))
                elif labels[i] == 1:
                    intraCluster[j] += (math.sqrt(
                        np.power(image_array[i][0] - centers[1][0], 2) + np.power(image_array[i][1] - centers[1][1],
                                                                                  2) + np.power(
                            image_array[i][2] - centers[1][2], 2)))
                elif labels[i] == 2:
                    intraCluster[j] += (math.sqrt(
                        np.power(image_array[i][0] - centers[2][0], 2) + np.power(image_array[i][1] - centers[2][1],
                                                                                  2) + np.power(
                            image_array[i][2] - centers[2][2], 2)))
                elif labels[i] == 3:
                    intraCluster[j] += (math.sqrt(
                        np.power(image_array[i][0] - centers[3][0], 2) + np.power(image_array[i][1] - centers[3][1],
                                                                                  2) + np.power(
                            image_array[i][2] - centers[3][2], 2)))
                elif labels[i] == 4:
                    intraCluster[j] += (math.sqrt(
                        np.power(image_array[i][0] - centers[4][0], 2) + np.power(image_array[i][1] - centers[4][1],
                                                                                  2) + np.power(
                            image_array[i][2] - centers[4][2], 2)))
                elif labels[i] == 5:
                    intraCluster[j] += (math.sqrt(
                        np.power(image_array[i][0] - centers[5][0], 2) + np.power(image_array[i][1] - centers[5][1],
                                                                                  2) + np.power(
                            image_array[i][2] - centers[5][2], 2)))
                elif labels[i] == 6:
                    intraCluster[j] += (math.sqrt(
                        np.power(image_array[i][0] - centers[6][0], 2) + np.power(image_array[i][1] - centers[6][1],
                                                                                  2) + np.power(
                            image_array[i][2] - centers[6][2], 2)))
                elif labels[i] == 7:
                    intraCluster[j] += (math.sqrt(
                        np.power(image_array[i][0] - centers[7][0], 2) + np.power(image_array[i][1] - centers[7][1],
                                                                                  2) + np.power(
                            image_array[i][2] - centers[7][2], 2)))
                elif labels[i] == 8:
                    intraCluster[j] += (math.sqrt(
                        np.power(image_array[i][0] - centers[8][0], 2) + np.power(image_array[i][1] - centers[8][1],
                                                                                  2) + np.power(
                            image_array[i][2] - centers[8][2], 2)))
                elif labels[i] == 9:
                    intraCluster[j] += (math.sqrt(
                        np.power(image_array[i][0] - centers[9][0], 2) + np.power(image_array[i][1] - centers[9][1],
                                                                                  2) + np.power(
                            image_array[i][2] - centers[9][2], 2)))

        else:
            labels = model.predict(image_array)
            centers = model.cluster_centers_
            for i in range(len(labels)):
                if labels[i] == 0:
                    intraCluster[j] += (math.sqrt(
                        np.power(image_array[i][0] - centers[0][0], 2) + np.power(image_array[i][1] - centers[0][1],
                                                                                  2) + np.power(
                            image_array[i][2] - centers[0][2], 2)))
                elif labels[i] == 1:
                    intraCluster[j] += (math.sqrt(
                        np.power(image_array[i][0] - centers[1][0], 2) + np.power(image_array[i][1] - centers[1][1],
                                                                                  2) + np.power(
                            image_array[i][2] - centers[1][2], 2)))
                elif labels[i] == 2:
                    intraCluster[j] += (math.sqrt(
                        np.power(image_array[i][0] - centers[2][0], 2) + np.power(image_array[i][1] - centers[2][1],
                                                                                  2) + np.power(
                            image_array[i][2] - centers[2][2], 2)))
                elif labels[i] == 3:
                    intraCluster[j] += (math.sqrt(
                        np.power(image_array[i][0] - centers[3][0], 2) + np.power(image_array[i][1] - centers[3][1],
                                                                                  2) + np.power(
                            image_array[i][2] - centers[3][2], 2)))
                elif labels[i] == 4:
                    intraCluster[j] += (math.sqrt(
                        np.power(image_array[i][0] - centers[4][0], 2) + np.power(image_array[i][1] - centers[4][1],
                                                                                  2) + np.power(
                            image_array[i][2] - centers[4][2], 2)))
                elif labels[i] == 5:
                    intraCluster[j] += (math.sqrt(
                        np.power(image_array[i][0] - centers[5][0], 2) + np.power(image_array[i][1] - centers[5][1],
                                                                                  2) + np.power(
                            image_array[i][2] - centers[5][2], 2)))
                elif labels[i] == 6:
                    intraCluster[j] += (math.sqrt(
                        np.power(image_array[i][0] - centers[6][0], 2) + np.power(image_array[i][1] - centers[6][1],
                                                                                  2) + np.power(
                            image_array[i][2] - centers[6][2], 2)))
                elif labels[i] == 7:
                    intraCluster[j] += (math.sqrt(
                        np.power(image_array[i][0] - centers[7][0], 2) + np.power(image_array[i][1] - centers[7][1],
                                                                                  2) + np.power(
                            image_array[i][2] - centers[7][2], 2)))
                elif labels[i] == 8:
                    intraCluster[j] += (math.sqrt(
                        np.power(image_array[i][0] - centers[8][0], 2) + np.power(image_array[i][1] - centers[8][1],
                                                                                  2) + np.power(
                            image_array[i][2] - centers[8][2], 2)))
                elif labels[i] == 9:
                    intraCluster[j] += (math.sqrt(
                        np.power(image_array[i][0] - centers[9][0], 2) + np.power(image_array[i][1] - centers[9][1],
                                                                                  2) + np.power(
                            image_array[i][2] - centers[9][2], 2)))

        print("done in %0.3fs." % (time() - t0))
        print(intraCluster[j])

        plt.figure(2)
        plt.clf()
        plt.axis('off')
        plt.title('Quantized image ({} colors, method={})'.format(n_colors, method))

        plt.imshow(recreate_image(centers, labels, rows, cols))
        plt.show()
    print(intraCluster)

    n_colors = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    plt.figure(3)
    plt.plot(n_colors, intraCluster, color="green", linewidth=1.0, )
    plt.title('Intra-cluster method={})'.format(method))
    plt.xlabel('Número de clusters')
    plt.ylabel('Suma de distancia entre los datos y su cluster')
    plt.show()


if __name__ == '__main__':
    image_name = input('Ingrese el nombre de la imagen: ')
    method = input('Ingrese el método: ')
    color_segmentation(image_name, method)