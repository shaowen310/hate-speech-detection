import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist

nltk.download("stopwords")
nltk.download("punkt")


def remove_stopwords(text, language="english"):
    stopwords = nltk.corpus.stopwords.words(language)
    return " ".join([w for w in text.split() if w not in stopwords])


def remove_punctuations_nltk(text, language="english"):
    """Don't remove negations."""
    l = nltk.word_tokenize(text, language=language)
    return " ".join(
        [w for w in l if not re.fullmatch("[" + string.punctuation + "]+", w)]
    )


def word_tokenize_word_freq(text):
    tokens = word_tokenize(text)
    freq_dist = FreqDist(tokens)
    return tokens, freq_dist


def doc_word_freq(texts, vocab_size=None):
    """A document contains a iterable of text strings.

    Parameters
    ----------
    vocab_size : int | None
        The number of words to keep in the vocabulary. `None` to keep all.
    """
    freq_dist_doc = FreqDist()
    for text in texts:
        tokens, freq_dist = word_tokenize_word_freq(text)
        freq_dist = FreqDist(tokens)
        freq_dist_doc.update(freq_dist)
    return freq_dist_doc.most_common(vocab_size)
