import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression,LinearRegression
import argparse
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
#import matplotlib.pyplot as plt

class MyLogisticRegression:
    def __init__(self, dataset_num, perform_test):
        self.training_set = None
        self.test_set = None
        self.model_logistic = None
        self.model_linear = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        
        self.perform_test = perform_test
        self.dataset_num = dataset_num
        self.read_csv(self.dataset_num)

    def read_csv(self, dataset_num):
        if dataset_num == '1':
            train_dataset_file = 'train_q1_1.csv'
            test_dataset_file = 'test_q1_1.csv'
        elif dataset_num == '2':
            train_dataset_file = 'train_q1_2.csv'
            test_dataset_file = 'test_q1_2.csv'
        else:
            print("unsupported dataset number")
            
        self.training_set = pd.read_csv(train_dataset_file, sep=',', header=0)

        if self.perform_test:
            self.test_set = pd.read_csv(test_dataset_file, sep=',', header=0)
            self.X_test = self.test_set[['exam_score_1' , 'exam_score_2']]
            self.y_test = self.test_set[['label']]        

        # format train data by column
        self.X_train = self.training_set[['exam_score_1' , 'exam_score_2']]
        self.y_train = self.training_set['label']
        
    def model_fit_linear(self):
        self.model_linear = LinearRegression(fit_intercept=True)
        self.model_linear.fit(self.X_train,self.y_train)
    
    def train_linear_model(self):
        self.read_csv(self.dataset_num)
        self.model_fit_linear()

    def train_logistic_model(self):
        self.read_csv(self.dataset_num)
        self.model_fit_logistic()
    
    def model_fit_logistic(self):
        self.model_logistic = LogisticRegression()
        self.model_logistic.fit(self.X_train , self.y_train)
    
    def model_predict_linear(self):
        self.train_linear_model()
        accuracy = 0.0
        precision, recall, f1, support = np.array([0,0]), np.array([0,0]), np.array([0,0]), np.array([0,0])
        assert self.model_linear is not None, "Initialize the model, i.e. instantiate the variable self.model_linear in model_fit_linear method"

        if self.X_train is not None:

            y_pred = self.model_linear.predict(self.X_test)
            y_pred_binary = (y_pred > 0.5).astype(bool)

            accuracy = self.model_linear.score(self.X_test , self.y_test.values.reshape(-1 , 1))
            precision , recall , f1 , support = precision_recall_fscore_support(self.y_test.values.reshape(-1 , 1) , y_pred_binary)   
        
        assert precision.shape == recall.shape == f1.shape == support.shape == (2,), "precision, recall, f1, support should be an array of shape (2,)"        
        return [accuracy, precision, recall, f1, support]

    def model_predict_logistic(self):
        self.train_logistic_model()
        accuracy = 0.0
        precision, recall, f1, support = np.array([0,0]), np.array([0,0]), np.array([0,0]), np.array([0,0])

        assert self.model_logistic is not None, "Initialize the model, i.e. instantiate the variable self.model_logistic in model_fit_logistic method"
        assert self.training_set is not None, "self.read_csv function isn't called or the self.trianing_set hasn't been initialized "

        if self.X_train is not None:
            y_pred = self.model_logistic.predict(self.X_test)
            accuracy = accuracy_score(self.y_test.values.reshape(-1 , 1) , y_pred)
            
            precision , recall , f1 , support = precision_recall_fscore_support(self.y_test.values.reshape(-1 , 1) , y_pred)
            
        
        assert precision.shape == recall.shape == f1.shape == support.shape == (2,), "precision, recall, f1, support should be an array of shape (2,)"
        return [accuracy, precision, recall, f1, support]


    '''
    def plot_graphs_linear(self):
        # exam 1 against exam 2 scores
        x1 = self.training_set['exam_score_1']
        x2 = self.training_set['exam_score_2']
        p1 , p2 = np.polyfit(x1 , x2 , 1)
        plt.plot(x2 , p1 * x2 + p2 , color='blue')
        plt.scatter(x1 , x2 , c=self.y_train)
        plt.xlabel("Exam 1 Scores")
        plt.ylabel("Exam 2 Scores")
        plt.title("Exam 1 vs. Exam 2 Scores (Linear - Dataset 2)")
        plt.show()

    def plot_graphs_logistic(self):
        # exam 1 against exam 2 scores
        x1 = self.training_set['exam_score_1'].values
        x2 = self.training_set['exam_score_2'].values

        # decision boundary
        x_min , x_max = x1.min() - 1 , x2.max() + 1
        y_min , y_max = x2.min() - 1 , x2.max() + 1
        xx , yy = np.meshgrid(np.arange(x_min , x_max , 0.01) ,
                              np.arange(y_min , y_max , 0.01))
        z = self.model_logistic.predict(pd.DataFrame(np.c_[xx.ravel() , yy.ravel()] , columns=['exam_score_1' , 'exam_score_2']))
        z = z.reshape(xx.shape)
        plt.contourf(xx , yy , z , alpha=0.8)       
        plt.scatter(x1 , x2 , c=self.y_train , edgecolors='k' , marker='o' , s=100)

        plt.xlabel("Exam 1 Scores")
        plt.ylabel("Exam 2 Scores")
        plt.title("Exam 1 vs. Exam 2 Scores (Logistic - Dataset 2)")
        plt.show()
    '''

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Linear Regression')
    parser.add_argument('-d','--dataset_num', type=str, default = "1", choices=["1","2"], help='string indicating datset number. For example, 1 or 2')
    parser.add_argument('-t','--perform_test', action='store_true', help='boolean to indicate inference')
    args = parser.parse_args()
    classifier = MyLogisticRegression(args.dataset_num, args.perform_test)
    acc = classifier.model_predict_linear()
    acc = classifier.model_predict_logistic()

    # plots
    #classifier.plot_graphs_linear()
    #classifier.plot_graphs_logistic()
    