import functools

from .text_process_utils import (
    remove_urls,
    remove_mentions,
    remove_hashtags,
    remove_punctuations,
    remove_numbers,
    remove_non_ascii,
    remove_extra_space,
)
from .text_process_nltk_utils import (
    word_tokenize_vocab,
    word_tokenize_word_freq,
    doc_word_freq,
    remove_stopwords,
)


def compose(*functions):
    return functools.reduce(lambda f, g: lambda x: f(g(x)), functions)


def transform_fn_text_cleanup():
    text_process_fn = compose(
        remove_extra_space,
        remove_punctuations,
        remove_numbers,
        remove_stopwords,
        remove_urls,
        remove_hashtags,
        remove_mentions,
        remove_non_ascii,
    )

    def transform_fn(samples):
        samples.loc[:, "text"] = samples["tweet"].apply(text_process_fn)
        return samples

    return transform_fn
