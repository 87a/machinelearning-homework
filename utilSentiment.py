import random
import sys
from collections import Counter

def generateClusteringExamples(numExamples, numWordsPerTopic, numFillerWords):
    '''
    Generate artificial examples inspired by sentiment for clustering.
    Each review has a hidden sentiment (positive or negative) and a topic (plot, acting, or music).
    The actual review consists of 2 sentiment words, 4 topic words and 1 filler word, for example:

        good:1 great:1 plot1:2 plot7:1 plot9:1 plot11:1 filler0:1

    numExamples: Number of examples to generate
    numWordsPerTopic: Number of words per topic (e.g., plot0, plot1, ...)
    numFillerWords: Number of words per filler (e.g., filler0, filler1, ...)
    '''
    sentiments = [['bad', 'awful', 'worst', 'terrible'],
                  ['good', 'great', 'fantastic', 'excellent']]
    topics = ['plot', 'acting', 'music']

    def generateExample():
        x = Counter()
        # Choose 2 sentiment words according to some sentiment
        sentimentWords = random.choice(sentiments)
        x[random.choice(sentimentWords)] += 1
        x[random.choice(sentimentWords)] += 1
        # Choose 4 topic words from a fixed topic
        topic = random.choice(topics)
        x[topic + str(random.randint(0, numWordsPerTopic - 1))] += 1
        x[topic + str(random.randint(0, numWordsPerTopic - 1))] += 1
        x[topic + str(random.randint(0, numWordsPerTopic - 1))] += 1
        x[topic + str(random.randint(0, numWordsPerTopic - 1))] += 1
        # Choose 1 filler word
        x['filler' + str(random.randint(0, numFillerWords - 1))] += 1
        return x

    random.seed(42)
    examples = [generateExample() for _ in range(numExamples)]
    return examples


def outputClusters(path, examples, centroids, assignments):
    '''
    Output the clusters to the given path.
    '''
    print('Outputting clusters to %s' % path)
    out = open(path, 'w', encoding='utf8')
    for j in range(len(centroids)):
        print('====== Cluster %s' % j, file=out)
        print('--- centroid:', file=out)
        for k, v in sorted(list(centroids[j].items()), key=lambda k_v: -k_v[1]):
            if v != 0:
                print('%s\t%s' % (k, v), file=out)
        print('--- Assigned points:', file=out)
        for i, z in enumerate(assignments):
            if z == j:
                print(' '.join(list(examples[i].keys())), file=out)
    out.close()


def EuclideanDistance_sparse(e0, e1):
    """
    @param dict e0: a feature vector represented by a mapping from a feature (string) to a weight (float).
    @param dict e1: same as e0
    @return float: the squared Euclidean distance between e0 and e1
    """
    return sum([ (e0.get(f, 0) - v) ** 2 for f, v in e1.items() ]) + \
           sum([ (e1.get(f, 0) - v) ** 2 for f, v in e0.items() if e1.get(f) is None ])


