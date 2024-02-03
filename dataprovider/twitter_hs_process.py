import functools

from .text_process_utils import (
    remove_urls,
    remove_mentions,
    remove_hashtags,
    remove_numbers,
    remove_non_ascii,
    remove_extra_space,
)
from .text_process_nltk_utils import (
    remove_stopwords,
    remove_punctuations_nltk,
)


def compose(*functions):
    return functools.reduce(lambda f, g: lambda x: f(g(x)), functions)


def transform_fn_text_cleanup():
    text_process_fn = compose(
        remove_extra_space,
        remove_punctuations_nltk,
        remove_numbers,
        remove_stopwords,
        remove_urls,
        remove_hashtags,
        remove_mentions,
        remove_non_ascii,
    )

    def transform_fn(samples):
        return samples.apply(text_process_fn)

    return transform_fn
