import json
from tqdm import tqdm
import random
import itertools
import gensim.models
import torch
import numpy as np

from utils import Corpus
from mapper_utils import train_AE_mapper
from VibeSpace import VibeSpace

if __name__ == "__main__":

    domain_names = ["song", "movie", "book"]

    sim_files = [f"similar_lists/song_7.txt",
                 "similar_lists/movie_8.txt",
                 "similar_lists/book_7.txt"]

    meta_files = [f"meta/{d}_meta.json" for d in domain_names]

    common_name_functions = [
        lambda met: f"{met['track']}-{met['artist']}",
        lambda met: f"{met['title']}",
        lambda met: f"{met['title']} - {met['isbn_10']}"
    ]

    mapping_files = {
        (p[0], p[1]): f"mappings/Mapping_from_{p[0]}_to_{p[1]}.json" for p in list(itertools.combinations(domain_names, 2))
    }
    mapping_files.update({
        (p[1], p[0]): f"mappings/Mapping_from_{p[1]}_to_{p[0]}.json" for p in list(itertools.combinations(domain_names, 2))
    })

    saved_vecspace_models = [
        "models/song.model",
        "models/movie.model",
        "models/book.model"
    ]

    vibespace = VibeSpace(domain_names=domain_names,
                          similarity_files=sim_files,
                          meta_files=meta_files,
                          common_name_functions=common_name_functions,
                          mapping_files=mapping_files,
                          similarity_metric="euclidean",
                         # embedding_size=1024,
                          embedding_size=1024,
                          train_epochs=100,
                          saved_vecspace_models=saved_vecspace_models,
                          #saved_vecspace_models=[],
                          load_mapper_weights=False)


    # AB_list = [
    #     ('book', 'movie'),
    #     ('book', 'song'),
    #     ('movie', 'book'),
    #     ('movie', 'song'),
    #     ('song', 'book'),
    #     ('song', 'movie')
    # ]
    # for ab in AB_list:
    #     print(ab)
    #     vibespace.aba_metric(ab[0], ab[1])
    #     vibespace.aba_metric_identity(ab[0], ab[1])

    ABC_list = [
        ('book', 'song', 'movie'),
        ('book', 'movie', 'song'),
        ('movie', 'song', 'book'),
        ('movie', 'book', 'song'),
        ('song', 'movie', 'book'),
        ('song', 'book', 'movie')
        ]

    for abc in ABC_list:
        print(abc)
        vibespace.abca_metric(abc[0], abc[1], abc[2])
        vibespace.abca_metric_identity(abc[0], abc[1], abc[2])

    print("DONE")