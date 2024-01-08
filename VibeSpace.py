import json
from tqdm import tqdm
import random
import itertools
import gensim.models
import torch
import numpy as np

from utils import Corpus, CosineSimilarity, EuclideanSimilarity
from mapper_utils import train_AE_mapper

"""
This script contains the VibeSpace class object.

VibeSpace object contains functionality for creating VibeSpaces given similar_list data, and for learning maps between
those spaces
"""

class VibeSpace(object):
    """
    It understands all vibes, it knows the essence of an entity.
    """
    def __init__(self,
                 domain_names: list,
                 similarity_files: list,
                 meta_files: list,
                 common_name_functions: list,
                 mapping_files: dict,
                 saved_vecspace_models: list = [],
                 similarity_metric: str = "cosine",
                 embedding_size: int=1000,
                 train_epochs: int=100,
                 learning_rate: float=0.001,
                 save_vecspaces: bool = False):
        """

        """
        self.embedding_size = embedding_size
        self.train_epochs = train_epochs
        self.learning_rate = learning_rate

        self.similarity_metric = similarity_metric
        if similarity_metric == "cosine":
            self.similarity_module = CosineSimilarity
        elif similarity_metric == "euclidean":
            self.similarity_module = EuclideanSimilarity
        else:
            print("ERROR: UNKNOWN SIM METRIC")
            exit(999)

        self.domain_names = domain_names
        self.mapping_files = {k: self._load_json(v) for k,v in mapping_files.items()}

        # Load all similarity and meta files
        self.sim_lists = {
            domain_names[i]: self._load_similarity_list(similarity_files[i]) for i in range(len(domain_names))
        }
        self.meta_files = {
            domain_names[i]: self._load_json(meta_files[i]) for i in range(len(domain_names))
        }

        self.common_name_functions = {domain_names[i]: common_name_functions[i] for i in range(len(domain_names))}

        # We must make vector embedding spaces for each domain
        if len(saved_vecspace_models) == 0:
            self.vibespaces = {
                d: self.make_vector_space(d) for d in domain_names
            }
            for k, v in self.vibespaces.items():
                v.save(f"models/{k}.model")
        else:
            self.vibespaces = {
                domain_names[i]: gensim.models.Word2Vec.load(saved_vecspace_models[i])for i in range(len(domain_names))
            }


        # We now make all the mappings
        self.mappings = {
            map_pair: self.make_mapping(map_pair) for map_pair in mapping_files.keys()
        }

        return

    # -----------------------------------------------------------------------------
    # Public Functions
    # -----------------------------------------------------------------------------

    # METRICS

    def aba_metric(self,
                   domain_a:str,
                   domain_b:str):
        assert domain_a in self.domain_names, "domain_a is not a known domain"
        assert domain_b in self.domain_names, "domain_b is not a known domain"
        similarities = []
        for e_a in tqdm(self.vibespaces[domain_a].wv.index_to_key[:1000]):
            vec_a = self.get_vector(domain_a, e_a)
            vec_b = self.map_entity(domain_a, domain_b, e_a)
            e_b, _ = self.get_most_similar_entity_from_vector(domain_b, vec_b)
            vec_a_dash = self.map_entity(domain_b, domain_a, e_b)
            e_a_dash, _ = self.get_most_similar_entity_from_vector(domain_a, vec_a_dash)
            vec_a_dash = self.get_vector(domain_a, e_a_dash)
            similarities.append(self.similarity_module.get_similarity(vec_a, vec_a_dash))
        print(f"METRIC RESULTS FOR {domain_a}->{domain_b}->{domain_a}")
        print(f"Average similarity between A and A' where A->B->A' is {np.mean(similarities)}")

    def abca_metric(self,
                   domain_a:str,
                   domain_b:str,
                   domain_c:str):
        assert domain_a in self.domain_names, "domain_a is not a known domain"
        assert domain_b in self.domain_names, "domain_b is not a known domain"
        assert domain_c in self.domain_names, "domain_c is not a known domain"
        similarities = []
        for e_a in tqdm(self.vibespaces[domain_a].wv.index_to_key[:1000]):
            vec_a = self.get_vector(domain_a, e_a)
            vec_b = self.map_entity(domain_a, domain_b, e_a)
            e_b, _ = self.get_most_similar_entity_from_vector(domain_b, vec_b)
            vec_c = self.map_entity(domain_b, domain_c, e_b)
            e_c, _ = self.get_most_similar_entity_from_vector(domain_c, vec_c)
            vec_a_dash = self.map_entity(domain_c, domain_a, e_c)
            e_a_dash, _ = self.get_most_similar_entity_from_vector(domain_a, vec_a_dash)
            vec_a_dash = self.get_vector(domain_a, e_a_dash)
            similarities.append(self.similarity_module.get_similarity(vec_a, vec_a_dash))
        print(f"METRIC RESULTS FOR {domain_a}->{domain_b}->{domain_c}->{domain_a}")
        print(f"Average similarity between A and A' where A->B->C->A' is {np.mean(similarities)}")

    def aba_metric_identity(self,
                            domain_a:str,
                            domain_b:str):
        assert domain_a in self.domain_names, "domain_a is not a known domain"
        assert domain_b in self.domain_names, "domain_b is not a known domain"
        similarities = []
        for e_a in tqdm(self.vibespaces[domain_a].wv.index_to_key[:1000]):
            vec_a = self.get_vector(domain_a, e_a)
            vec_b = vec_a
            e_b, _ = self.get_most_similar_entity_from_vector(domain_b, vec_b)
            vec_a_dash = self.get_vector(domain_b, e_b)
            e_a_dash, _ = self.get_most_similar_entity_from_vector(domain_a, vec_a_dash)
            vec_a_dash = self.get_vector(domain_a, e_a_dash)
            similarities.append(self.similarity_module.get_similarity(vec_a, vec_a_dash))
        print(f"IDENTITY METRIC RESULTS FOR {domain_a}->{domain_b}->{domain_a}")
        print(f"IDENTITY: Average similarity between A and A' where A->B->A' is {np.mean(similarities)}")

    def abca_metric_identity(self,
                            domain_a:str,
                            domain_b:str,
                            domain_c:str):
        assert domain_a in self.domain_names, "domain_a is not a known domain"
        assert domain_b in self.domain_names, "domain_b is not a known domain"
        assert domain_c in self.domain_names, "domain_c is not a known domain"
        similarities = []
        for e_a in tqdm(self.vibespaces[domain_a].wv.index_to_key[:1000]):
            vec_a = self.get_vector(domain_a, e_a)
            vec_b = vec_a
            e_b, _ = self.get_most_similar_entity_from_vector(domain_b, vec_b)
            vec_c = self.get_vector(domain_b, e_b)
            e_c, _ = self.get_most_similar_entity_from_vector(domain_c, vec_c)
            vec_a_dash = self.get_vector(domain_c, e_c)
            e_a_dash, _ = self.get_most_similar_entity_from_vector(domain_a, vec_a_dash)
            vec_a_dash = self.get_vector(domain_a, e_a_dash)
            similarities.append(self.similarity_module.get_similarity(vec_a, vec_a_dash))
        print(f"IDENTITY METRIC RESULTS FOR {domain_a}->{domain_b}->{domain_c}->{domain_a}")
        print(f"IDENTITY: Average similarity between A and A' where A->B->C->A' is {np.mean(similarities)}")

    # NON - METRICS

    def make_mapping(self,
                   map_pair):
        mapping = self.mapping_files[map_pair]
        frm = map_pair[0]
        to = map_pair[1]
        filtered_mapping = {}
        for k, map_list in tqdm(mapping.items()):
            kdash = k.replace("_", " ")
            if kdash in self.meta_files[frm].keys() and isinstance(self.meta_files[frm][kdash], dict):
                common_name = self.common_name_functions[frm](self.meta_files[frm][kdash])
                if common_name in self.vibespaces[frm].wv.index_to_key:
                    if map_list == [] or isinstance(map_list[0], dict):
                        continue
                    mapped_entities = [m for m in map_list if
                                       m.lower() in self.meta_files[to].keys() and isinstance(self.meta_files[to][m.lower()],
                                                                                  dict) and self.common_name_functions[to](
                                           self.meta_files[to][m.lower()]) in self.vibespaces[to].wv.index_to_key]
                    if mapped_entities == []:
                        pass
                        # print(f"No mapped entities for {k} ({domain_from}) are in meta_dict for domain {domain_to}")
                    else:
                        filtered_mapping[common_name] = [self.common_name_functions[to](self.meta_files[to][m.lower()]) for m in mapped_entities]
                        # print(f"{common_name} maps to {filtered_mapping[common_name]}")
                else:
                    pass

        dataset = []
        for k, v_list in tqdm(filtered_mapping.items()):
            if k in self.vibespaces[frm].wv.index_to_key:
                k_vec = self.vibespaces[frm].wv[self.vibespaces[frm].wv.key_to_index[k]]
                vecs = []
                for v in v_list:
                    v_vec = self.vibespaces[to].wv[self.vibespaces[to].wv.key_to_index[v]]
                    vecs.append(v_vec)
                dataset.append(
                    {
                        "input": k_vec,
                        "input_name": k,
                        "targets": vecs,
                        "target_names": v_list
                    }
                )
            else:
                print(f"{k} doesn't work.")
                exit(1)

        mapper = train_AE_mapper(dataset,
                                 num_epochs=self.train_epochs,
                                 loss_func=self.similarity_metric,
                                 learning_rate=self.learning_rate)
        mapper.eval()
        return mapper


    def make_vector_space(self,
                          domain: str,
                          num_permutations: int = 0):
        print(f"Making vector space for domain {domain}...")
        if num_permutations > 0:
            print("num_permutations is above zero, so list permutations will be automatically added (this will inflate the dataset)")
        sentence_list = self.sim_lists[domain]
        meta = self.meta_files[domain]
        final_sentence_list = []
        failed = []
        for sent in tqdm(sentence_list):
            ents = sent.split("//")
            filtered_ents = []
            for e in ents:
                if e in meta.keys():
                    if isinstance(meta[e], dict):
                        filtered_ents.append(
                            self.common_name_functions[domain](meta[e])
                        )
                else:
                    print(f"{e} not in meta dict. Skipping")
                    failed.append(e)
            for I in range(2, min(len(ents), 11)):
                these_ents = filtered_ents[:I]
                final_sentence_list.append("//".join(these_ents))
                if num_permutations > 0:
                    for p_i in range(num_permutations):
                        p = random.sample(these_ents, len(these_ents))
                        final_sentence_list.append("//".join(p))
        print(f"Final corpus is {len(final_sentence_list)} sentences long")
        print(f"There were {len(failed)} failures")
        corpus = Corpus(final_sentence_list)
        model = gensim.models.Word2Vec(sentences=corpus, min_count=10,
                                       vector_size=self.embedding_size, window=6)
        return model

    def get_vector(self, domain, entity):
        return self.vibespaces[domain].wv.vectors[self.vibespaces[domain].wv.key_to_index[entity]]

    def get_dataset_size(self, domain):
        """
        ..todo:: this isn't quite informative because we filter some of the entities away e.g. if they are not in the meta files or are suspected inventions
        """
        entity_list = []
        for sent in self.sim_lists[domain]:
            entity_list += sent.split("//")
        entity_list = sorted(list(set(entity_list)))
        print(f"Corpus is {len(self.sim_lists[domain])} sentences long")
        print(f"There are {len(entity_list)} unique entities...")
        return len(self.sim_lists[domain]), len(entity_list)

    def map_entity(self,
                   domain_from,
                   domain_to,
                   entity):
        # We need to find the film in the filtered mapping which is most similar to the given prompt
        query_vector = self.vibespaces[domain_from].wv[entity]
        return self.map_vector(domain_from, domain_to, query_vector)

    def map_vector(self,
                   domain_from,
                   domain_to,
                   vector):
        if isinstance(vector, np.ndarray):
            vector = torch.tensor(vector)
        mapped_vector = self.mappings[(domain_from, domain_to)](vector)
        return mapped_vector

    def map_entity_chain(self,
                         entity: str,
                         domain_chain: list):
        assert len(domain_chain) > 1
        vector_chain = []
        mapped_vector = self.map_entity(domain_chain[0], domain_chain[1], entity)
        vector_chain.append(mapped_vector)
        if len(domain_chain) > 2:
            for i in range(2, len(domain_chain)):
                mapped_vector = self.map_vector(domain_chain[i-1], domain_chain[i], mapped_vector)
                vector_chain.append(mapped_vector)
        return mapped_vector, vector_chain

    def get_most_similar_entity_from_vector(self,
                                            domain: str,
                                            vector: np.ndarray):
        most_sims, sims = self.get_most_similar_entities_from_vector(domain, vector)
        return most_sims[0], sims[0]

    def get_most_similar_entities_from_vector(self,
                                              domain: str,
                                              vector):
        if torch.is_tensor(vector):
            vector = vector.detach().cpu().numpy()
        similarities = self.similarity_module.get_similarities(vector, self.vibespaces[domain].wv.vectors)
        isort = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)
        similar_entities = np.array(self.vibespaces[domain].wv.index_to_key)[np.array(isort)]
        similarity_scores = np.array(similarities)[isort]
        return similar_entities, similarity_scores

    def print_most_similar_entities_from_vector(self,
                                              domain: str,
                                              vector: np.ndarray,
                                              ):
        similar_entities, similarity_scores = self.get_most_similar_entities_from_vector(domain, vector)
        print(f"Most similar {domain}s:")
        for i in range(5):
            print(f"{similar_entities[i]}: {similarity_scores[i]}")
        print("\n")
        print(f"Most dissimilar {domain}s:")
        for i in range(5):
            print(f"{similar_entities[-i - 1]}: {similarity_scores[-i - 1]}")
        print("\n")

    def run(self):
        search_dicts = {}
        for d in self.domain_names:
            search_dicts[d] = {
                s: s.replace("_", " ").replace("|", " ").replace("(", "").replace(")", "").lower().strip().split(" ") for s
                in self.vibespaces[d].wv.index_to_key}
        while True:
            dchain = input("What domain chain would you like to try?\n")
            dchain = dchain.split(", ")
            failed = False
            for d in dchain:
                if d not in self.domain_names:
                    print("Invalid domain chain idiot, try again loser!")
                    failed = True
                    break
            if failed:
                continue
            search_complete = False
            while not search_complete:
                sterm = input("Which entity would you like to search for?\n")
                if sterm in self.vibespaces[dchain[0]].wv.index_to_key:
                    end_vec, vec_chain = self.map_entity_chain(entity=sterm,
                                                               domain_chain=dchain)


                    similarities = self.similarity_module.get_similarities(end_vec.detach().cpu().numpy(), self.vibespaces[dchain[-1]].wv.vectors)

                    isort = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)
                    similar_entities = np.array(self.vibespaces[dchain[-1]].wv.index_to_key)[np.array(isort)]
                    similarity_scores = np.array(similarities)[isort]

                    print(f"Most similar {dchain[-1]}s:")
                    for i in range(5):
                        print(f"{similar_entities[i]}: {similarity_scores[i]}")
                    print("\n")
                    print(f"Most dissimilar {dchain[-1]}s:")
                    for i in range(5):
                        print(f"{similar_entities[-i - 1]}: {similarity_scores[-i - 1]}")
                    print("\n")
                    search_complete = True
                else:
                    shared = {s: len(set(sterm.split(" ")) & set(search_dicts[dchain[0]][s])) for s in search_dicts[dchain[0]].keys()};
                    sim_sterms = '\n'.join(sorted(shared, key=shared.get, reverse=True)[:10])
                    print(f"No entity matching that string, most similar are...\n{sim_sterms}")

    # -----------------------------------------------------------------------------
    # Private Functions
    # -----------------------------------------------------------------------------

    def _load_similarity_list(self, path):
        with open(path, "r", encoding="utf8") as f:
            corpus = f.read()
        sentence_list = [lst for lst in corpus.lower().split("\n")]
        #..todo:: is below really necessary?
        #sentence_list = [e for e in sentence_list if "similar" not in "".join(e)]
        return sentence_list

    def _load_json(self, path):
        with open(path) as f:
            js = json.load(f)
        return js

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

    saved_vecspace_models = [f"models/{d}.model" for d in ["song", "movie", "book"]]

    vibespace = VibeSpace(domain_names=domain_names,
                          similarity_files=sim_files,
                          meta_files=meta_files,
                          common_name_functions=common_name_functions,
                          similarity_metric="euclidean",
                          #similarity_metric="cosine",
                          embedding_size=1024,
                          train_epochs=100,
                          saved_vecspace_models=[],
                          #saved_vecspace_models=saved_vecspace_models,
                          mapping_files=mapping_files)

    vibespace.run()