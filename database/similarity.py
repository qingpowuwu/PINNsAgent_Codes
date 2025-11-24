# database/similarity.py

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def cosine_sim(a, b):
    return cosine_similarity([a], [b])[0][0]