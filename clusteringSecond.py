from utilSentiment import *
import numpy as np


def kmeans(examples, k):
    # This is just a toy. You have to implement the kmeans algorithm to replace this toy.
    centroids = [i for i in examples]
    random.shuffle(centroids)
    centroids = centroids[:k]  # 随机初始化中心
    # centroids, assignments = examples, list(range(len(examples)))
    assignments = np.zeros(len(examples))  # 初始化Z
    EPOCHS = 10000
    for epoch in range(EPOCHS):
        centroids_ = []
        for i in range(0, len(examples)):  # 找到最近的中心
            mincentroids = 0
            for k_ in range(k):
                dist = EuclideanDistance_sparse(examples[i], centroids[k_])
                if k_ == 0:
                    minDist = dist
                # print(dist)
                if dist < minDist:
                    minDist = dist
                    mincentroids = k_
            assignments[i] = mincentroids
        # print(assignments)

        for k_ in range(k):  # 求新的中心
            count = 0
            templist = []
            keys = []
            for z in range(len(examples)):
                if assignments[z] == k_:
                    count += 1
                    templist.append(examples[z])
            for i in templist:
                for j in i.keys():
                    keys.append(j)
            keys = list(set(keys))
            centroid_ = Counter()
            for key in keys:
                keysum = 0
                for i in templist:
                    keysum += i[key]
                centroid_[key] = keysum / count
            centroids_.append(centroid_)
        # print(centroids_)
        if centroids_ == centroids:
            print(epoch)
            break
        else:
            centroids = centroids_
        # c = (centroids_ == centroids)
        # if c.all():
        #     break
        # else:
        #     centroids = centroids_
    return centroids, assignments


if __name__ == '__main__':
    # This is just a toy. You should the test your code with a variety of settings for parameters, inlcuding
    # numExamples, numWordsPerTopic, numFillerWords, and the k.

    dataset = generateClusteringExamples(10000, 30, 5)
    # print(dataset)
    # print(EuclideanDistance_sparse(dataset[0], dataset[1]))
    # random.shuffle(dataset)
    # centroids = dataset[:3]
    # print(centroids)
    # print(list(range(len(dataset))))
    centroids, assignments = kmeans(dataset, 3)
    outputClusters('mySecondClusters.txt', dataset, centroids, assignments)
