"""TF-IDF keyword engine — focus signals for the coach, not if/else rules."""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer


class TfidfEngine:
    """
    Builds importance scores over conversation texts.
    Returns top keywords for the latest user turn in context of history.
    """

    def __init__(self, top_k: int = 6) -> None:
        self.top_k = top_k
        self._vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
            max_features=500,
        )

    def extract(self, history: list[str], latest: str) -> list[str]:
        corpus = [t for t in history + [latest] if t and t.strip()]
        if not corpus:
            return []

        # Single short utterance: still fit — TF alone surfaces content words
        matrix = self._vectorizer.fit_transform(corpus)
        feature_names = self._vectorizer.get_feature_names_out()
        latest_row = matrix[-1].toarray().ravel()

        ranked = sorted(
            zip(feature_names, latest_row, strict=True),
            key=lambda pair: pair[1],
            reverse=True,
        )
        return [word for word, score in ranked[: self.top_k] if score > 0]
