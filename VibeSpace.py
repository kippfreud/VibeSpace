import itertools
import json
import random

from collections import Counter
import gensim.models
import matplotlib
import numpy as np
import torch
import umap
from sklearn.manifold import TSNE
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score

matplotlib.use("TkAgg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import seaborn as sns

from mapper_utils import train_AE_mapper
from utils import Corpus, CosineSimilarity, EuclideanSimilarity

"""
This script contains the VibeSpace class object.

VibeSpace object contains functionality for creating VibeSpaces given similar_list data, and for learning maps between
those spaces
"""


class VibeSpace(object):
    """
    It understands all vibes, it knows the essence of an entity.
    """

    def __init__(
        self,
        domain_names: list,
        similarity_files: list,
        meta_files: list,
        common_name_functions: list,
        mapping_files: dict,
        saved_vecspace_models: list = [],
        load_mapper_weights: bool = False,
        similarity_metric: str = "cosine",
        embedding_size: int = 1000,
        train_epochs: int = 100,
        learning_rate: float = 0.001,
        num_permutations: int = 0,
        save_vecspaces: bool = False,
        device="cpu",
    ):
        """ """
        self.device = device
        self.embedding_size = embedding_size
        self.train_epochs = train_epochs
        self.learning_rate = learning_rate
        self.load_mapper_weights = load_mapper_weights

        self.similarity_metric = similarity_metric
        if similarity_metric == "cosine":
            self.similarity_module = CosineSimilarity
        elif similarity_metric == "euclidean":
            self.similarity_module = EuclideanSimilarity
        else:
            print("ERROR: UNKNOWN SIM METRIC")
            exit(999)

        self.domain_names = domain_names
        self.mapping_files = {k: self._load_json(v) for k, v in mapping_files.items()}

        # Load all similarity and meta files
        self.sim_lists = {
            domain_names[i]: self._load_similarity_list(similarity_files[i])
            for i in range(len(domain_names))
        }
        self.meta_files = {
            domain_names[i]: self._load_json(meta_files[i])
            for i in range(len(domain_names))
        }

        self.common_name_functions = {
            domain_names[i]: common_name_functions[i] for i in range(len(domain_names))
        }
        self.common_to_ent = {}

        # We must make vector embedding spaces for each domain
        if len(saved_vecspace_models) == 0:
            self.vibespaces = {
                d: self.make_vector_space(d, num_permutations=num_permutations)
                for d in domain_names
            }
            for k, v in self.vibespaces.items():
                v.save(f"models/{k}.model")
        else:
            try:
                self.vibespaces = {
                    self.make_vector_space(
                        d, num_permutations=num_permutations, from_path=saved_vecspace_models[i]
                    )
                    for d, i in enumerate(domain_names)
                }
            except:
                print(
                    "WARNING: Saved vector models were given, but could not be loaded. Making new vecspaces."
                )
                self.vibespaces = {
                    d: self.make_vector_space(d, num_permutations=num_permutations)
                    for d in domain_names
                }
                for k, v in self.vibespaces.items():
                    v.save(f"models/{k}.model")

        # UMAP dimensionality reduction
        # self.reduced_vibespaces = {d: umap.UMAP(
        #     n_neighbors=5,
        #     min_dist=0.001
        # ).fit_transform(self.vibespaces[d].wv.vectors) for d in domain_names}
        # TSNE dimensionality reduction
        self.reduced_vibespaces = {
            d: TSNE(
                n_components=2,
                learning_rate="auto",
                init="random",
                perplexity=1.0,
                random_state=42,
            ).fit_transform(self.vibespaces[d].wv.vectors)
            for d in domain_names
        }
        # No dim reduction
        # self.reduced_vibespaces = {d: self.vibespaces[d].wv.vectors for d in domain_names}

        # self.plot_reduced_maps()

        # We now make all the mappings
        self.mappings = {
            map_pair: self.make_mapping(map_pair) for map_pair in mapping_files.keys()
        }

        return

    # -----------------------------------------------------------------------------
    # Public Functions
    # -----------------------------------------------------------------------------

    def plot_reduced_maps(self):
        for d in self.domain_names:
            plt.scatter(
                self.reduced_vibespaces[d][:, 0],
                self.reduced_vibespaces[d][:, 1],
            )
            plt.gca().set_aspect("equal", "datalim")
            plt.title(f"UMAP projection of the {d} domain", fontsize=24)
            plt.show()

    def f7(self, seq):
        seen = set()
        seen_add = seen.add
        return [x for x in seq if not (x in seen or seen_add(x))]

    def visualize_mapping(self, domain_chain, entity_chain, title=None):
        assert len(domain_chain) == len(
            entity_chain
        ), "Entity chain and domain chain must be the same length"
        domains = self.f7(domain_chain)
        num_domains = len(domains)
        index_chain = [
            self.vibespaces[domain_chain[i]].wv.key_to_index[entity_chain[i]]
            for i in range(len(entity_chain))
        ]
        if num_domains > 3:
            print("ERROR: I don't know how to plot this many domains...")
            exit(1)
        if num_domains == 1:
            print("WARNING: Plotting for one domain is not implemented properly yet...")
            plt.scatter(
                self.reduced_vibespaces[domains[0]][:, 0],
                self.reduced_vibespaces[domains[0]][:, 1],
                c="lightblue",
                alpha=1,
                s=2,
            )
            plt.show()
            return
        plotmaps = {
            domains[0]: self.reduced_vibespaces[domains[0]],
            domains[1]: self.reduced_vibespaces[domains[1]]
            + np.array(
                [2.0 * np.max(self.reduced_vibespaces[domains[0]]), 0]
            ),  # mp.max(self.reduced_vibespaces[domains[0]])[0]
        }
        # plt.scatter(plotmaps[domains[0]][:,0], plotmaps[domains[0]][:,1], c="green")
        plt.scatter(
            plotmaps[domains[1]][:, 0],
            plotmaps[domains[1]][:, 1],
            c="lightblue",
            alpha=1,
            s=2,
        )
        plt.axis("off")
        if num_domains == 3:
            plotmaps[domains[2]] = self.reduced_vibespaces[domains[2]] + np.array(
                [
                    1.0 * np.max(self.reduced_vibespaces[domains[0]]),
                    1.5 * np.max(self.reduced_vibespaces[domains[0]]),
                ]
            )
            plt.scatter(
                plotmaps[domains[2]][:, 0],
                plotmaps[domains[2]][:, 1],
                c="violet",
                alpha=1,
                s=2,
            )

        for dnI, dname in enumerate(plotmaps.keys()):
            avp = np.mean(plotmaps[dname], axis=0)
            if dnI < 2:
                y = -128
            else:
                y = 300
                avp[0] = avp[0] - 15
            print(dname)
            plt.text(
                avp[0] - 3,
                y,
                dname,
                rotation=0,
                fontsize=24,
            )  # fontweight="bold")
        sim_rank = np.flip(
            np.argsort(
                [
                    self.similarity_module.get_similarity(
                        self.vibespaces[domain_chain[0]].wv.vectors[index_chain[0]], v
                    )
                    for v in self.vibespaces[domain_chain[0]].wv.vectors
                ]
            )
        )
        cmap = np.linspace(1, 0, len(sim_rank))  # [sim_rank]
        # plt.scatter(plotmaps[domain_chain[0]][sim_rank[:]][:, 0], plotmaps[domain_chain[0]][sim_rank[:]][:, 1], c=cmap[:])
        plt.scatter(
            plotmaps[domain_chain[0]][sim_rank[:]][:, 0],
            plotmaps[domain_chain[0]][sim_rank[:]][:, 1],
            c="lightgreen",
            alpha=1,
            s=2,
        )
        arrowpatches = []
        for i in range(1, len(domain_chain)):
            start_xy = plotmaps[domain_chain[i - 1]][index_chain[i - 1]]
            if i == 1:
                plt.text(
                    start_xy[0] + 5.5 - 70,
                    start_xy[1] - 0.5 + 9.5 - 40,
                    entity_chain[i - 1].split("-")[0],
                    fontsize=18,
                )  # fontweight="bold")
                pass
            elif i == 2:
                plt.text(
                    start_xy[0] - 5 + 10,
                    start_xy[1] - 5,
                    entity_chain[i - 1].split("-")[0],
                    fontsize=18,
                )  # fontweight="bold")
                pass
            elif i == 3:
                plt.text(
                    start_xy[0] - 80,
                    start_xy[1] + 10,
                    entity_chain[i - 1].split("-")[0],
                    fontsize=18,
                )  # fontweight="bold")
            else:
                plt.text(
                    start_xy[0] - 80,
                    start_xy[1] + 10,
                    entity_chain[i - 1].split("-")[0],
                    fontsize=18,
                )  # fontweight="bold")
            end_xy = plotmaps[domain_chain[i]][index_chain[i]]
            dxdy = end_xy - start_xy
            # connectionstyle = patches.ConnectionStyle("Arc3", rad=0.2)
            connectionstyle = "arc3,rad=-0.3"
            cheat = 0.0
            if i == 1:
                # connectionstyle = patches.ConnectionStyle("Arc3", rad=-0.2)
                connectionstyle = "arc3,rad=0.3"
                cheat = 0.25
            if i == 2:
                # connectionstyle = patches.ConnectionStyle("Arc3", rad=0.2)
                connectionstyle = "arc3,rad=0.3"
            # plt.arrow(start_xy[0], start_xy[1], dxdy[0], dxdy[1], width=0.2,
            #           color="black", linestyle="-", head_starts_at_zero=True)
            arrowpatches.append(
                patches.FancyArrowPatch(
                    (start_xy[0], start_xy[1] - cheat),
                    (start_xy[0] + dxdy[0], start_xy[1] + dxdy[1]),
                    connectionstyle=connectionstyle,
                    mutation_scale=10.0,
                    color="black",
                )
            )
            plt.scatter([start_xy[0]], [start_xy[1]], c="pink", s=5)

        plt.text(
            end_xy[0] - 118, end_xy[1] - 5, entity_chain[-1].split("-")[0], fontsize=18
        )  # , fontweight="bold")
        plt.scatter([end_xy[0]], [end_xy[1]], c="red", s=5)
        if title is not None:
            plt.title(title)
        for patch in arrowpatches:
            plt.gca().add_patch(patch)
            # pass
        plt.show()

    # -----------------------------------------------------------------------------
    # Plotting Functions
    # -----------------------------------------------------------------------------

    def get_similarity_percentile(
        self, domain_a: str, vec_a: np.ndarray, vec_a_dash: np.ndarray
    ):
        sims = [
            self.similarity_module.get_similarity(vec_a, v)
            for v in self.vibespaces[domain_a].wv.vectors
        ]
        s_star = self.similarity_module.get_similarity(vec_a, vec_a_dash)
        percentile = len([d for d in sims if d < s_star]) / len(sims)
        return percentile

    # METRICS
    def aba_metric(self, domain_a: str, domain_b: str):
        assert domain_a in self.domain_names, "domain_a is not a known domain"
        assert domain_b in self.domain_names, "domain_b is not a known domain"
        similarities = []
        percentiles = []
        # for e_a in tqdm(self.vibespaces[domain_a].wv.index_to_key[:1000]):
        for e_a_i in tqdm(range(100)):
            e_a = self.vibespaces[domain_a].wv.index_to_key[e_a_i]
            vec_a = self.get_vector(domain_a, e_a)
            vec_b = self.map_entity(domain_a, domain_b, e_a)
            e_b, _ = self.get_most_similar_entity_from_vector(domain_b, vec_b)
            vec_a_dash = self.map_entity(domain_b, domain_a, e_b)
            e_a_dash, _ = self.get_most_similar_entity_from_vector(domain_a, vec_a_dash)
            vec_a_dash = self.get_vector(domain_a, e_a_dash)
            sim = self.similarity_module.get_similarity(vec_a, vec_a_dash)
            similarities.append(sim)
            per = self.get_similarity_percentile(domain_a, vec_a, vec_a_dash)
            percentiles.append(per)
            title = f"Entity {e_a} -> {e_b} -> {e_a_dash} is closer than {per:.2f}% of other {domain_a}'s: similarity={sim:.2f}"
            self.visualize_mapping(
                (domain_a, domain_b, domain_a), (e_a, e_b, e_a_dash), title=title
            )
        print(f"METRIC RESULTS FOR {domain_a}->{domain_b}->{domain_a}")
        print(
            f"Average similarity between A and A' where A->B->A' is {np.mean(percentiles)}"
        )

    def abca_metric(self, domain_a: str, domain_b: str, domain_c: str):
        assert domain_a in self.domain_names, "domain_a is not a known domain"
        assert domain_b in self.domain_names, "domain_b is not a known domain"
        assert domain_c in self.domain_names, "domain_c is not a known domain"
        similarities = []
        percentiles = []
        for e_a_i in tqdm(range(100)):
            e_a = self.vibespaces[domain_a].wv.index_to_key[e_a_i]
            vec_a = self.get_vector(domain_a, e_a)
            vec_b = self.map_entity(domain_a, domain_b, e_a)
            e_b, _ = self.get_most_similar_entity_from_vector(domain_b, vec_b)
            vec_c = self.map_entity(domain_b, domain_c, e_b)
            e_c, _ = self.get_most_similar_entity_from_vector(domain_c, vec_c)
            vec_a_dash = self.map_entity(domain_c, domain_a, e_c)
            e_a_dash, _ = self.get_most_similar_entity_from_vector(domain_a, vec_a_dash)
            vec_a_dash = self.get_vector(domain_a, e_a_dash)
            sim = self.similarity_module.get_similarity(vec_a, vec_a_dash)
            similarities.append(sim)
            per = self.get_similarity_percentile(domain_a, vec_a, vec_a_dash)
            percentiles.append(per)
            title = f"Entity {e_a} -> {e_b} -> {e_c} -> {e_a_dash} is closer than {per*100: .2f}% of other {domain_a}'s"
            # self.visualize_mapping((domain_a, domain_b, domain_c, domain_a), (e_a, e_b, e_c, e_a_dash), title=title)
        print(f"METRIC RESULTS FOR {domain_a}->{domain_b}->{domain_c}->{domain_a}")
        print(
            f"Average similarity between A and A' where A->B->C->A' is {np.mean(percentiles)}"
        )

    def aba_metric_identity(self, domain_a: str, domain_b: str):
        assert domain_a in self.domain_names, "domain_a is not a known domain"
        assert domain_b in self.domain_names, "domain_b is not a known domain"
        similarities = []
        percentiles = []
        for e_a_i in tqdm(range(100)):
            e_a = self.vibespaces[domain_a].wv.index_to_key[e_a_i]
            vec_a = self.get_vector(domain_a, e_a)
            vec_b = vec_a
            e_b, _ = self.get_most_similar_entity_from_vector(domain_b, vec_b)
            vec_a_dash = self.get_vector(domain_b, e_b)
            e_a_dash, _ = self.get_most_similar_entity_from_vector(domain_a, vec_a_dash)
            vec_a_dash = self.get_vector(domain_a, e_a_dash)
            similarities.append(
                self.similarity_module.get_similarity(vec_a, vec_a_dash)
            )
            per = self.get_similarity_percentile(domain_a, vec_a, vec_a_dash)
            percentiles.append(per)
            # print(f"Entity {e_a} -> {e_b} -> {e_a_dash} is closer than {per}% of other {domain_a}'s")
        print(f"IDENTITY METRIC RESULTS FOR {domain_a}->{domain_b}->{domain_a}")
        print(
            f"IDENTITY: Average similarity between A and A' where A->B->A' is {np.mean(percentiles)}"
        )

    def abca_metric_identity(self, domain_a: str, domain_b: str, domain_c: str):
        assert domain_a in self.domain_names, "domain_a is not a known domain"
        assert domain_b in self.domain_names, "domain_b is not a known domain"
        assert domain_c in self.domain_names, "domain_c is not a known domain"
        similarities = []
        percentiles = []
        for e_a_i in tqdm(range(100)):
            e_a = self.vibespaces[domain_a].wv.index_to_key[e_a_i]
            vec_a = self.get_vector(domain_a, e_a)
            vec_b = vec_a
            e_b, _ = self.get_most_similar_entity_from_vector(domain_b, vec_b)
            vec_c = self.get_vector(domain_b, e_b)
            e_c, _ = self.get_most_similar_entity_from_vector(domain_c, vec_c)
            vec_a_dash = self.get_vector(domain_c, e_c)
            e_a_dash, _ = self.get_most_similar_entity_from_vector(domain_a, vec_a_dash)
            vec_a_dash = self.get_vector(domain_a, e_a_dash)
            similarities.append(
                self.similarity_module.get_similarity(vec_a, vec_a_dash)
            )
            per = self.get_similarity_percentile(domain_a, vec_a, vec_a_dash)
            percentiles.append(per)
            # print(f"Entity {e_a} -> {e_b} -> {e_c} -> {e_a_dash} is closer than {per}% of other {domain_a}'s")
        print(
            f"IDENTITY METRIC RESULTS FOR {domain_a}->{domain_b}->{domain_c}->{domain_a}"
        )
        print(
            f"IDENTITY: Average similarity between A and A' where A->B->C->A' is {np.mean(percentiles)}"
        )

    def aba_metric_mean(self, domain_a: str, domain_b: str):
        assert domain_a in self.domain_names, "domain_a is not a known domain"
        assert domain_b in self.domain_names, "domain_b is not a known domain"
        similarities = []
        percentiles = []
        for e_a_i in tqdm(range(100)):
            e_a = self.vibespaces[domain_a].wv.index_to_key[e_a_i]
            vec_a = self.get_vector(domain_a, e_a)
            vec_b = np.mean(self.vibespaces[domain_b].wv.vectors, axis=0)
            e_b, _ = self.get_most_similar_entity_from_vector(domain_b, vec_b)
            vec_a_dash = np.mean(self.vibespaces[domain_a].wv.vectors, axis=0)
            e_a_dash, _ = self.get_most_similar_entity_from_vector(domain_a, vec_a_dash)
            vec_a_dash = self.get_vector(domain_a, e_a_dash)
            similarities.append(
                self.similarity_module.get_similarity(vec_a, vec_a_dash)
            )
            per = self.get_similarity_percentile(domain_a, vec_a, vec_a_dash)
            percentiles.append(per)
            # print(f"Entity {e_a} -> {e_b} -> {e_a_dash} is closer than {per}% of other {domain_a}'s")
        print(f"MEAN METRIC RESULTS FOR {domain_a}->{domain_b}->{domain_a}")
        print(
            f"MEAN: Average similarity between A and A' where A->B->A' is {np.mean(percentiles)}"
        )

    def abca_metric_mean(self, domain_a: str, domain_b: str, domain_c: str):
        assert domain_a in self.domain_names, "domain_a is not a known domain"
        assert domain_b in self.domain_names, "domain_b is not a known domain"
        assert domain_c in self.domain_names, "domain_c is not a known domain"
        similarities = []
        percentiles = []
        for e_a_i in tqdm(range(100)):
            e_a = self.vibespaces[domain_a].wv.index_to_key[e_a_i]
            vec_a = self.get_vector(domain_a, e_a)
            vec_b = np.mean(self.vibespaces[domain_b].wv.vectors, axis=0)
            e_b, _ = self.get_most_similar_entity_from_vector(domain_b, vec_b)
            vec_c = np.mean(self.vibespaces[domain_c].wv.vectors, axis=0)
            e_c, _ = self.get_most_similar_entity_from_vector(domain_c, vec_c)
            vec_a_dash = np.mean(self.vibespaces[domain_a].wv.vectors, axis=0)
            e_a_dash, _ = self.get_most_similar_entity_from_vector(domain_a, vec_a_dash)
            vec_a_dash = self.get_vector(domain_a, e_a_dash)
            similarities.append(
                self.similarity_module.get_similarity(vec_a, vec_a_dash)
            )
            per = self.get_similarity_percentile(domain_a, vec_a, vec_a_dash)
            percentiles.append(per)
            # print(f"Entity {e_a} -> {e_b} -> {e_c} -> {e_a_dash} is closer than {per}% of other {domain_a}'s")
        print(
            f"IDENTITY METRIC RESULTS FOR {domain_a}->{domain_b}->{domain_c}->{domain_a}"
        )
        print(
            f"IDENTITY: Average similarity between A and A' where A->B->C->A' is {np.mean(percentiles)}"
        )

    # NON - METRICS

    def make_mapping(self, map_pair):
        print(map_pair)
        mapping = self.mapping_files[map_pair]
        frm = map_pair[0]
        to = map_pair[1]
        filtered_mapping = {}
        for k, map_list in tqdm(mapping.items()):
            kdash = k.replace("_", " ")
            if kdash in self.meta_files[frm].keys() and isinstance(
                self.meta_files[frm][kdash], dict
            ):
                common_name = self.common_name_functions[frm](
                    self.meta_files[frm][kdash]
                )
                if common_name in self.vibespaces[frm].wv.index_to_key:
                    if map_list == [] or isinstance(map_list[0], dict):
                        continue
                    mapped_entities = [
                        m
                        for m in map_list
                        if m.lower() in self.meta_files[to].keys()
                        and isinstance(self.meta_files[to][m.lower()], dict)
                        and self.common_name_functions[to](
                            self.meta_files[to][m.lower()]
                        )
                        in self.vibespaces[to].wv.index_to_key
                    ]
                    if mapped_entities == []:
                        pass
                        # print(f"No mapped entities for {k} ({domain_from}) are in meta_dict for domain {domain_to}")
                    else:
                        filtered_mapping[common_name] = [
                            self.common_name_functions[to](
                                self.meta_files[to][m.lower()]
                            )
                            for m in mapped_entities
                        ]
                        # print(f"{common_name} maps to {filtered_mapping[common_name]}")
                else:
                    pass

        dataset = []
        for k, v_list in tqdm(filtered_mapping.items()):
            if k in self.vibespaces[frm].wv.index_to_key:
                k_vec = self.vibespaces[frm].wv[self.vibespaces[frm].wv.key_to_index[k]]
                vecs = []
                for v in v_list:
                    v_vec = self.vibespaces[to].wv[
                        self.vibespaces[to].wv.key_to_index[v]
                    ]
                    vecs.append(v_vec)
                dataset.append(
                    {
                        "input": k_vec,
                        "input_name": k,
                        "targets": vecs,
                        "target_names": v_list,
                    }
                )
            else:
                print(f"{k} doesn't work.")
                exit(1)

        mapper = train_AE_mapper(
            dataset,
            num_epochs=self.train_epochs,
            loss_func=self.similarity_metric,
            learning_rate=self.learning_rate,
            model_path=f"mapping_models/{map_pair[0]}_to_{map_pair[1]}.pt",
            load_map=self.load_mapper_weights,
            device=self.device,
        )
        mapper.eval()
        return mapper

    def make_vector_space(self, domain: str, num_permutations: int = 0, from_path=None):
        print(f"Making vector space for domain {domain}...")
        if num_permutations > 0:
            print(
                "num_permutations is above zero, so list permutations will be automatically added (this will inflate the dataset)"
            )
        sentence_list = self.sim_lists[domain]
        meta = self.meta_files[domain]
        self.common_to_ent[domain] = {}
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
                        if e not in self.common_to_ent[domain].keys():
                            self.common_to_ent[domain][self.common_name_functions[domain](meta[e])] = e
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
        if from_path is None:
            corpus = Corpus([f for f in final_sentence_list if f!=""])
            model = gensim.models.Word2Vec(
                sentences=corpus, min_count=10, vector_size=self.embedding_size, window=6
            )
        else:
            model = gensim.models.Word2Vec.load(from_path)
        return model

    def get_vector(self, domain, entity):
        return self.vibespaces[domain].wv.vectors[
            self.vibespaces[domain].wv.key_to_index[entity]
        ]

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

    def map_entity(self, domain_from, domain_to, entity):
        # We need to find the film in the filtered mapping which is most similar to the given prompt
        query_vector = self.vibespaces[domain_from].wv[entity]
        return self.map_vector(domain_from, domain_to, query_vector)

    def map_vector(self, domain_from, domain_to, vector):
        if isinstance(vector, np.ndarray):
            vector = torch.tensor(vector).to(self.device)
        mapped_vector = self.mappings[(domain_from, domain_to)](vector)
        return mapped_vector

    def map_entity_chain(self, entity: str, domain_chain: list):
        if len(domain_chain) > 1:
            vector_chain = [torch.tensor(self.get_vector(domain_chain[0], entity))]
            mapped_vector = self.map_entity(domain_chain[0], domain_chain[1], entity)
            mapped_vector = self.get_vector(
                domain_chain[1],
                self.get_most_similar_entity_from_vector(
                    domain_chain[1], mapped_vector
                )[0],
            )
            vector_chain.append(torch.tensor(mapped_vector))
            if len(domain_chain) > 2:
                for i in range(2, len(domain_chain)):
                    mapped_vector = self.map_vector(
                        domain_chain[i - 1], domain_chain[i], mapped_vector
                    )
                    mapped_vector = self.get_vector(
                        domain_chain[i],
                        self.get_most_similar_entity_from_vector(
                            domain_chain[i], mapped_vector
                        )[0],
                    )
                    vector_chain.append(mapped_vector)
            return mapped_vector, vector_chain
        else:
            vector_chain = [
                torch.tensor(self.get_vector(domain_chain[0], entity)),
                torch.tensor(self.get_vector(domain_chain[0], entity)),
            ]
            ret = self.get_most_similar_entities_from_vector(
                domain_chain[0], vector_chain[0]
            )
            return vector_chain[0], vector_chain

    def get_most_similar_entity_from_vector(self, domain: str, vector: np.ndarray):
        most_sims, sims = self.get_most_similar_entities_from_vector(domain, vector)
        return most_sims[0], sims[0]

    def get_most_similar_entities_from_vector(self, domain: str, vector):
        if torch.is_tensor(vector):
            vector = vector.detach().cpu().numpy()
        similarities = self.similarity_module.get_similarities(
            vector, self.vibespaces[domain].wv.vectors
        )
        isort = sorted(
            range(len(similarities)), key=lambda i: similarities[i], reverse=True
        )
        similar_entities = np.array(self.vibespaces[domain].wv.index_to_key)[
            np.array(isort)
        ]
        similarity_scores = np.array(similarities)[isort]
        return similar_entities, similarity_scores

    def print_most_similar_entities_from_vector(
        self,
        domain: str,
        vector: np.ndarray,
    ):
        similar_entities, similarity_scores = (
            self.get_most_similar_entities_from_vector(domain, vector)
        )
        print(f"Most similar {domain}s:")
        for i in range(5):
            print(f"{similar_entities[i]}: {similarity_scores[i]}")
        print("\n")
        print(f"Most dissimilar {domain}s:")
        for i in range(5):
            print(f"{similar_entities[-i - 1]}: {similarity_scores[-i - 1]}")
        print("\n")

    def generate_1vsRest(self, domain, meta, n=10, min_tag_count=1):
        print(f"\nGenerating 1vsRest for {domain}...")
        tags = [self.meta_files[domain][self.common_to_ent[domain][n]][meta] for n in self.vibespaces[domain].wv.index_to_key]
        tags = [t if t is not None else [] for t in tags]
        tag_count = dict(Counter([item for sublist in tags for item in sublist]).most_common())
        valid_tags = [k for k, v in tag_count.items() if v > min_tag_count]
        tags = [[t for t in T if t in valid_tags] for T in tags]
        has_tags = [t != [] for t in tags]
        x = self.vibespaces[domain].wv.vectors[has_tags]
        all_tags = list(itertools.chain(*tags))
        common_tags = Counter(all_tags).most_common(n)
        for c in common_tags:
            y = np.array([c[0] in t for t in tags])[has_tags]
            reg = SVC(kernel='linear',
                      class_weight='balanced',
                      #C=10.
                      ) #also try non-linear
            reg.fit(x, y)
            yhat = reg.predict(x)
            f1 = f1_score(y, yhat)
            acc = accuracy_score(y, yhat)
            print(f"For {meta} {c[0]} (size {sum(y)}):\n\tf1 = {f1}\n\tacc = {acc}")

    def run(self):
        search_dicts = {}
        for d in self.domain_names:
            search_dicts[d] = {
                s: s.replace("_", " ")
                .replace("|", " ")
                .replace("(", "")
                .replace(")", "")
                .lower()
                .strip()
                .split(" ")
                for s in self.vibespaces[d].wv.index_to_key
            }
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
                    end_vec, vec_chain = self.map_entity_chain(
                        entity=sterm, domain_chain=dchain
                    )
                    end_vec = torch.tensor(end_vec).to(self.device)
                    similarities = self.similarity_module.get_similarities(
                        end_vec.detach().cpu().numpy(),
                        self.vibespaces[dchain[-1]].wv.vectors,
                    )

                    isort = sorted(
                        range(len(similarities)),
                        key=lambda i: similarities[i],
                        reverse=True,
                    )
                    similar_entities = np.array(
                        self.vibespaces[dchain[-1]].wv.index_to_key
                    )[np.array(isort)]
                    similarity_scores = np.array(similarities)[isort]

                    print(f"Most similar {dchain[-1]}s:")
                    for i in range(5):
                        print(f"{similar_entities[i]}: {similarity_scores[i]}")
                    print("\n")
                    print(f"Most dissimilar {dchain[-1]}s:")
                    for i in range(5):
                        print(
                            f"{similar_entities[-i - 1]}: {similarity_scores[-i - 1]}"
                        )
                    print("\n")
                    if len(dchain) > 1:
                        ent_chain = [
                            self.get_most_similar_entity_from_vector(
                                dchain[i], vec_chain[i]
                            )[0]
                            for i in range(len(vec_chain))
                        ]
                    else:
                        dchain = [dchain[0], dchain[0]]
                        ent_chain = [
                            self.get_most_similar_entity_from_vector(
                                dchain[0], vec_chain[0]
                            )[0],
                            self.get_most_similar_entities_from_vector(
                                dchain[1], vec_chain[1]
                            )[0][1],
                        ]
                    self.visualize_mapping(dchain, ent_chain)
                    search_complete = True
                else:
                    shared = {
                        s: len(set(sterm.split(" ")) & set(search_dicts[dchain[0]][s]))
                        for s in search_dicts[dchain[0]].keys()
                    }
                    sim_sterms = "\n".join(
                        sorted(shared, key=shared.get, reverse=True)[:10]
                    )
                    print(
                        f"No entity matching that string, most similar are...\n{sim_sterms}"
                    )

    # -----------------------------------------------------------------------------
    # Private Functions
    # -----------------------------------------------------------------------------

    def _load_similarity_list(self, path):
        with open(path, "r", encoding="utf8") as f:
            corpus = f.read()
        sentence_list = [lst for lst in corpus.lower().split("\n")]
        # ..todo:: is below really necessary?
        # sentence_list = [e for e in sentence_list if "similar" not in "".join(e)]
        return sentence_list

    def _load_json(self, path):
        print(path)
        with open(path) as f:
            js = json.load(f)
        return js


if __name__ == "__main__":

    domain_names = ["song", "movie", "book"]

    sim_files = [
        f"similar_lists/song_7.txt",
        "similar_lists/movie_8.txt",
        "similar_lists/book_7.txt",
    ]

    meta_files = [f"meta/{d}_meta.json" for d in domain_names]

    common_name_functions = [
        lambda met: f"{met['track']}-{met['artist']}",
        lambda met: f"{met['title']}",
        lambda met: f"{met['title']} - {met['isbn_10']}",
    ]

    mapping_files = {
        (p[0], p[1]): f"mappings/Mapping_from_{p[0]}_to_{p[1]}.json"
        for p in list(itertools.combinations(domain_names, 2))
    }
    mapping_files.update(
        {
            (p[1], p[0]): f"mappings/Mapping_from_{p[1]}_to_{p[0]}.json"
            for p in list(itertools.combinations(domain_names, 2))
        }
    )

    saved_vecspace_models = [f"models/{d}.model" for d in ["song", "movie", "book"]]

    # saved_map_weights = [f"mapping_models/{d1}_to_{d2}.pt" for d1 in ["song", "movie", "book"] for ]

    vibespace = VibeSpace(
        domain_names=domain_names,
        similarity_files=sim_files,
        meta_files=meta_files,
        common_name_functions=common_name_functions,
        similarity_metric="cosine",
        embedding_size=1028,
        train_epochs=100,
        # saved_vecspace_models=[],
        saved_vecspace_models=saved_vecspace_models,
        load_mapper_weights=True,
        mapping_files=mapping_files,
    )

    vibespace.generate_1vsRest("book", "subjects", min_tag_count=900)
    # # vibespace.generate_1vsRest("book", "ol_genre", min_tag_count=1000)
    vibespace.generate_1vsRest("song", "tags", min_tag_count=1000)
    vibespace.generate_1vsRest("movie", "genres", min_tag_count=1000)

    vibespace.run()
