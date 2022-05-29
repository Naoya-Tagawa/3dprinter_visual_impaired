import numpy as np 
import matplotlib.pyplot as plt
#matplotlib inline
from sklearn.neighbors import NearestNeighbors 
import time

X = np.array([[2.1, 1.3], [1.3, 3.2], [2.9, 2.5], [2.7, 5.4], [3.8, 0.9], 
              [7.3, 2.1], [4.2, 6.5], [3.8, 3.7], [2.5, 4.1], [3.4, 1.9], 
              [5.7, 3.5], [6.1, 4.3], [5.1, 2.2], [6.2, 1.1]]) 
X1 = np.array([[18.,53.],[107.,142.]])
plt.figure() 
plt.title('Input data') 
plt.scatter(X1[:,0], X1[:,1], marker='o', s=75, color='black') 
plt.show()
k = 5 
test_datapoint = [4.3, 2.7] 
test_datapoint = [108.,126.]
start = time.perf_counter()
knn_model = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(X1) 
distances, indices = knn_model.kneighbors([test_datapoint]) 

print("K Nearest Neighbors:") 
for rank, index in enumerate(indices[0][:k], start=1): 
    print(str(rank) + " ==>", X1[index]) 
end = time.perf_counter()
print(end-start)

plt.figure() 
plt.title('Nearest neighbors') 
plt.scatter(X[:, 0], X[:, 1], marker='o', s=75, color='k') 
plt.scatter(X[indices][0][:][:, 0], X[indices][0][:][:, 1], 
            marker='o', s=250, color='k', facecolors='none') 
plt.scatter(test_datapoint[0], test_datapoint[1], 
            marker='x', s=75, color='k') 
plt.show() 
