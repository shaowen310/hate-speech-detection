import nltk
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist

nltk.download("stopwords")


def remove_stopwords(text, language="english"):
    stopwords = nltk.corpus.stopwords.words(language)
    return [word for word in text if word not in stopwords]


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


def word_tokenize_vocab(text, vocab, unk_token="<unk>"):
    tokens = word_tokenize(text)
    limited_tokens = [word if word in vocab else unk_token for word in tokens]
    return limited_tokens
