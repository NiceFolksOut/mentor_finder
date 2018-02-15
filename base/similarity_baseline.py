
import pickle
import wmd

import numpy as np


class SimilarutyRepo(object):

    def __init__(self):
        with open("nbow_500k.pickle", "rb") as fin:
            self.nbow = pickle.load(fin)

        with open("id2vec_500k.pickle", "rb") as fin:
            _, _, self.embeddings = pickle.load(fin)

    def similarity(self, **kwargs):
        embeddings = np.array(self.embeddings, dtype=np.float32)
        repo_index = {r[0]: i for i, r in enumerate(self.nbow)}

        class nbowobj(object):
            def __iter__(self):
                return iter(range(len(self.nbow)))

            def __getitem__(self, key):
                r = self.nbow[key]
                pairs = r[1]
                words = np.array([p[0] for p in pairs], dtype=np.int32)
                weights = np.array([p[1] for p in pairs], dtype=np.float32)
                return r[0], words, weights

        nnwmd = wmd.WMD(embeddings, nbowobj())
        nnwmd.cache_centroids()

        kwargs = dict(skipped_stop=0.9, early_stop=0.2, max_time=180)
        nearest_repos = nnwmd.nearest_neighbors(
            repo_index["dlang/dmd"], 10 + 1, **kwargs
        )

        return [self.nbow[n[0]][0] for n in nearest_repos]
