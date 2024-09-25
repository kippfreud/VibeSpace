from typing import List

import numpy as np
from numpy import dot
from numpy.linalg import norm


def _cos_sim(a, b):
    return dot(a, b) / (norm(a) * norm(b))


def _euclidean_dist(a, b):
    dist = norm(a - b)
    return -dist


class Similarity_Module:
    def __init__(self, similarity_function):
        self.similarity_function = similarity_function

    def get_similarities(self, A, L):
        return [self.similarity_function(A, vector) for vector in L]

    def get_similarity(self, a, b):
        return self.similarity_function(a, b)

    def average_sim(self, vectors: List[np.ndarray]) -> float:
        """
        Calculate the average similarity between all pairs of vectors in a list.

        :param vectors: A list of numpy arrays representing the vectors.
        :return: The average Euclidean distance between the vectors.
        """
        if len(vectors) < 2:
            raise ValueError("At least two vectors are required to compute distances.")

        # Calculate the sum of all pairwise Euclidean distances
        distance_sum = 0
        num_pairs = 0
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                distance_sum += self.similarity_function(vectors[i], vectors[j])
                num_pairs += 1

        # Calculate the average distance
        average_distance = distance_sum / num_pairs
        return average_distance


CosineSimilarity = Similarity_Module(_cos_sim)
EuclideanSimilarity = Similarity_Module(_euclidean_dist)


class Corpus:
    """An iterator that yields sentences (lists of str)."""

    def __init__(self, sentence_corpus, seperator="//"):
        self._corpus = sentence_corpus
        self._sep = seperator

    def __iter__(self):
        for line in self._corpus:
            ret = line.split(self._sep)
            yield ret
