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


def remove_twitter_word_user(text):
    """The word "user" is just a placeholder for the actual user names and it occurs many times without providing any useful information."""
    return re.sub(r"(?<=\s)user(?=\s)", "", text)
