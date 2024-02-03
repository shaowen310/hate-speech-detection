import re


def remove_urls(text):
    return re.sub(r"http\S+", "", text)


def remove_mentions(text):
    return re.sub(r"@\w+", "", text)


def remove_hashtags(text):
    return re.sub(r"#\w+", "", text)


def remove_punctuations(text):
    return re.sub(r"[^\w\s]", "", text)


def remove_numbers(text):
    return re.sub(r"\d+", "", text)


def remove_words_fn(words):
    def remove_fn(text):
        return " ".join([w for w in text.split() if w not in words])

    return remove_fn


def remove_non_ascii(text):
    return text.encode("ascii", "ignore").decode()


def remove_extra_space(text):
    return re.sub(r"\s+", " ", text.strip())
