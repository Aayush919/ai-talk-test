"""Tiny TF-IDF smoke — no API keys needed."""

from core.tfidf_engine import TfidfEngine


def main() -> None:
    engine = TfidfEngine(top_k=5)
    history = [
        "Hello I want to practice English",
        "I am preparing for a job interview",
    ]
    latest = "Can we practice interview questions about teamwork?"
    print(engine.extract(history, latest))


if __name__ == "__main__":
    main()
