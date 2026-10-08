"""Worker functions for parallel LDA inference (kept in their own module so spawned processes import it cheaply)."""

from gensim import corpora
from gensim.models import LdaModel

from src.topics import tokenize

_LDA = _DIC = None


def init():
    """Load the topic model once in each worker process."""
    global _LDA, _DIC
    _LDA = LdaModel.load("models/lda_model.gensim")
    _DIC = corpora.Dictionary.load("models/lda_dictionary.gensim")


def infer(texts):
    """Give each text its most likely topic. Returns [(topic, probability)], with (-1, 0.0) if no usable words."""
    out = []
    for t in texts:
        bow = _DIC.doc2bow(tokenize(t))
        if not bow:
            out.append((-1, 0.0))
            continue
        k, p = max(_LDA.get_document_topics(bow, minimum_probability=0), key=lambda x: x[1])
        out.append((int(k), float(p)))
    return out
