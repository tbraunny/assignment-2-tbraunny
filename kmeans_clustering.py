import numpy as np
import argparse
import scipy.io
from sklearn.cluster import KMeans
#import matplotlib.pyplot as plt

class MykmeansClustering:
    def __init__(self, dataset_file):
        self.model = None
        self.mat = None
        
        self.dataset_file = dataset_file
        self.read_mat()

    def read_mat(self):
        mat = scipy.io.loadmat(self.dataset_file)
        #print(mat.keys())

        self.data = mat['X']
        #print(self.data)
        
    def model_fit(self):
        self.model = KMeans(3)
        self.model.fit(self.data)
        
        cluster_centers = np.array([[0,0]])

        cluster_centers = self.model.cluster_centers_
        cluster_labels = self.model.labels_

        # plot cluster centers
        '''
        x = cluster_centers[:,0]
        y = cluster_centers[:,1]
        x_data = self.data[:,0]
        y_data = self.data[:,1]
        
        plt.scatter(x_data , y_data , c=cluster_labels)
        plt.scatter(x , y , color='red' , marker='X')
        plt.title('Cluster Centers')
        plt.show()
        '''

        return cluster_centers

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Kmeans clustering')
    parser.add_argument('-d','--dataset_file', type=str, default = "dataset_q2.mat", help='path to dataset file')
    args = parser.parse_args()
    classifier = MykmeansClustering(args.dataset_file)
    clusters_centers = classifier.model_fit()
    