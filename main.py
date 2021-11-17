
from kmeans import KMeans
from sklearn.datasets import make_blobs
from matplotlib import pyplot as plt

def main():
    kmeans = KMeans(K=5)
    data = make_blobs(10000, n_features=2, centers=5, cluster_std=0.5)
    print(data)
    kmeans.fit(data[0])
    labels = kmeans.predict(data[0])
    plt.scatter(data[0][:,0], data[0][:,1], c=labels)
    plt.savefig("graph")
    

if __name__ == "__main__":
    main()