import numpy as np
import argparse
import scipy.io
from sklearn.cluster import KMeans

class MykmeansClustering:
    def __init__(self, dataset_file):
        self.model = None
        
        self.dataset_file = dataset_file
        self.read_mat()

    def read_mat(self):
        mat = scipy.io.loadmat(self.dataset_file)
        
    def model_fit(self):
        '''
        initialize self.model here and execute kmeans clustering here
        '''
        
        cluster_centers = np.array([[0,0]])
        return cluster_centers

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Kmeans clustering')
    parser.add_argument('-d','--dataset_file', type=str, default = "dataset_q2.mat", help='path to dataset file')
    args = parser.parse_args()
    classifier = MykmeansClustering(args.dataset_file)
    clusters_centers = classifier.model_fit()
    print(clusters_centers)
    