import logging

from http.server import HTTPServer, SimpleHTTPRequestHandler, test
import os
import shutil
import threading
import time
from modelforge import generate_meta
from modelforge.model import Model, split_strings, write_model, merge_strings
import numpy


class DocumentFrequencies(Model):
    NAME = "docfreq"

    def construct(self, docs, tokens, freqs):
        self._docs = docs
        self._log.info("Building the docfreq dictionary...")
        self._df = dict(zip(tokens, freqs))
        return self

    def _load_tree(self, tree):
        self.construct(docs=tree["docs"], tokens=split_strings(tree["tokens"]),
                       freqs=tree["freqs"])

    def dump(self):
        return """Number of words: %d
First 10 words: %s
Number of documents: %d""" % (
            len(self._df), self.tokens()[:10], self.docs)

    @property
    def docs(self):
        return self._docs

    def prune(self, threshold: int):
        self._log.info("Pruning to min %d occurrences", threshold)
        pruned = DocumentFrequencies()
        pruned._docs = self.docs
        pruned._df = {k: v for k, v in self._df.items() if v >= threshold}
        pruned._meta = self.meta
        return pruned

    def __getitem__(self, item):
        return self._df[item]

    def __iter__(self):
        return iter(self._df.items())

    def __len__(self):
        return len(self._df)

    def get(self, item, default):
        return self._df.get(item, default)

    def tokens(self):
        return sorted(self._df)

    def save(self, output, deps=None):
        if not deps:
            deps = tuple()
        self._meta = generate_meta(self.NAME, 0, *deps)
        tokens = self.tokens()
        freqs = numpy.array([self._df[t] for t in tokens], dtype=numpy.float32)
        if tokens:
            write_model(self._meta,
                        {"docs": self.docs,
                         "tokens": merge_strings(tokens),
                         "freqs": freqs},
                        output)
        else:
            self._log.warning("Did not write %s because the model is empty", output)


class CORSWebServer(object):
    def __init__(self):
        self.thread = None
        self.server = None

    def serve(self):
        outer = self

        class ClojureServer(HTTPServer):
            def __init__(self, *args, **kwargs):
                HTTPServer.__init__(self, *args, **kwargs)
                outer.server = self

        class CORSRequestHandler(SimpleHTTPRequestHandler):
            def end_headers(self):
                self.send_header("Access-Control-Allow-Origin", "*")
                SimpleHTTPRequestHandler.end_headers(self)

        test(CORSRequestHandler, ClojureServer)

    def start(self):
        self.thread = threading.Thread(target=self.serve)
        self.thread.start()

    def stop(self):
        if self.running:
            self.server.shutdown()
            self.server.server_close()
            self.thread.join()
            self.server = None
            self.thread = None

    @property
    def running(self):
        return self.server is not None


web_server = CORSWebServer()


def present_embeddings(destdir, run_server, labels, index, embeddings):
    if not os.path.isdir(destdir):
        os.makedirs(destdir)
    os.chdir(destdir)
    metaf = "id2vec_meta.tsv"
    with open(metaf, "w") as fout:
        if len(labels) > 1:
            fout.write("\t".join(labels) + "\n")
        for item in index:
            if len(labels) > 1:
                fout.write("\t".join(item) + "\n")
            else:
                fout.write(item + "\n")
    dataf = "id2vec_data.tsv"
    with open(dataf, "w") as fout:
        for vec in embeddings:
            fout.write("\t".join(str(v) for v in vec))
            fout.write("\n")
    jsonf = "id2vec.json"
    with open(jsonf, "w") as fout:
        fout.write("""{
  "embeddings": [
    {
      "tensorName": "id2vec",
      "tensorShape": [%s, %s],
      "tensorPath": "http://0.0.0.0:8000/%s",
      "metadataPath": "http://0.0.0.0:8000/%s"
    }
  ]
}
""" % (len(embeddings), len(embeddings[0]), dataf, metaf))
    if run_server and not web_server.running:
        web_server.start()
    url = "http://projector.tensorflow.org/?config=http://0.0.0.0:8000/" + jsonf
    if run_server:
        if shutil.which("xdg-open") is not None:
            os.system("xdg-open " + url)
        else:
            browser = os.getenv("BROWSER", "")
            if browser:
                os.system(browser + " " + url)
            else:
                print("\t" + url)


def wait():
    log = logging.getLogger("projector")
    secs = int(os.getenv("PROJECTOR_SERVER_TIME", "60"))
    log.info("Sleeping for %d seconds, safe to Ctrl-C" % secs)
    try:
        time.sleep(secs)
    except KeyboardInterrupt:
        pass
    web_server.stop()


class Id2Vec(Model):
    NAME = "id2vec"

    def construct(self, embeddings, tokens):
        self._embeddings = embeddings
        self._tokens = tokens
        self._log.info("Building the token index...")
        self._token2index = {w: i for i, w in enumerate(self._tokens)}
        return self

    def _load_tree(self, tree):
        self.construct(embeddings=tree["embeddings"].copy(),
                       tokens=split_strings(tree["tokens"]))

    def dump(self):
        return """Shape: %s
First 10 words: %s""" % (
            self.embeddings.shape, self.tokens[:10])

    @property
    def embeddings(self):
        return self._embeddings

    @property
    def tokens(self):
        return self._tokens

    def items(self):
        return self._token2index.items()

    def __getitem__(self, item):
        return self._token2index[item]

    def __len__(self):
        return len(self._tokens)

    def save(self, output, deps=None):
        if not deps:
            deps = tuple()
        self._meta = generate_meta(self.NAME, 0, *deps)
        write_model(self._meta,
                    {"embeddings": self.embeddings,
                     "tokens": merge_strings(self.tokens)},
                    output)


def projector_entry(args):
    MAX_TOKENS = 10000  # hardcoded in Tensorflow Projector
    id2vec = Id2Vec(log_level=args.log_level).load(source=args.input)
    if args.df:
        df = DocumentFrequencies(log_level=args.log_level).load(source=args.df)
    else:
        df = None
    if len(id2vec) < MAX_TOKENS:
        tokens = numpy.arange(len(id2vec), dtype=int)
        if df is not None:
            freqs = [df.get(id2vec.tokens[i], 0) for i in tokens]
        else:
            freqs = None
    else:
        if df is not None:
            items = []
            for token, idx in id2vec.items():
                try:
                    items.append((df[token], idx))
                except KeyError:
                    continue
            items.sort(reverse=True)
            tokens = [i[1] for i in items[:MAX_TOKENS]]
            freqs = [i[0] for i in items[:MAX_TOKENS]]
        else:
            numpy.random.seed(777)
            tokens = numpy.random.choice(
                numpy.arange(len(id2vec), dtype=int), MAX_TOKENS,
                replace=False)
            freqs = None
    embeddings = numpy.vstack([id2vec.embeddings[i] for i in tokens])
    tokens = [id2vec.tokens[i] for i in tokens]
    labels = ["subtoken"]
    if freqs is not None:
        labels.append("docfreq")
        tokens = list(zip(tokens, (str(i) for i in freqs)))
    present_embeddings(args.output, not args.no_browser, labels,
                       tokens, embeddings)
    if not args.no_browser:
        wait()
