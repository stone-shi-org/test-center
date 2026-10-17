"""Built-in evaluation corpus and queries for embedding benchmarks.

Contains 30 diverse documents across 10 domains and 30 paired retrieval queries
with verified ground-truth matches for measuring Recall@k and MRR.
"""

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class DocumentItem:
    """Document in the evaluation corpus."""
    id: str
    text: str
    topic: str
    index: int


@dataclass(frozen=True)
class QueryItem:
    """Evaluation query with ground-truth document target."""
    id: str
    text: str
    expected_doc_id: str
    expected_index: int
    topic: str


RAW_DOCUMENTS: list[tuple[str, str]] = [
    ("The quick brown fox jumps over the lazy dog.", "animals"),
    ("Elephants are the largest land animals and have remarkable memories.", "animals"),
    ("Dolphins use echolocation to navigate and hunt in murky water.", "animals"),
    ("Photosynthesis converts sunlight into chemical energy in plants.", "biology"),
    ("DNA carries the genetic instructions for the development of living organisms.", "biology"),
    ("Black holes have gravitational fields so strong that not even light can escape.", "physics"),
    ("The capital of France is Paris, located on the Seine river.", "geography"),
    ("The Great Wall of China stretches over thirteen thousand miles.", "geography"),
    ("Mount Everest is the tallest mountain above sea level on Earth.", "geography"),
    ("Python is a popular programming language for data science.", "technology"),
    ("Neural networks are inspired by the human brain's structure.", "technology"),
    ("Blockchain technology enables decentralized and tamper-resistant record keeping.", "technology"),
    ("Regular exercise improves cardiovascular health and mood.", "health"),
    ("A balanced diet rich in vegetables supports long-term health.", "health"),
    ("Sleep is essential for memory consolidation and immune function.", "health"),
    ("Water boils at one hundred degrees Celsius at sea level pressure.", "physics"),
    ("Atoms consist of protons, neutrons, and electrons.", "physics"),
    ("Light behaves as both a particle and a wave depending on how it's observed.", "physics"),
    ("The Eiffel Tower was constructed in eighteen eighty-nine.", "history"),
    ("The Roman Empire fell in the fifth century after centuries of expansion.", "history"),
    ("The printing press revolutionized the spread of information in Europe.", "history"),
    ("Sushi originated in Japan and traditionally features raw fish and rice.", "food"),
    ("Coffee beans are roasted to develop their characteristic aroma and flavor.", "food"),
    ("Fermentation is used to make foods like yogurt, kimchi, and bread.", "food"),
    ("Soccer is the most widely played sport across the world.", "sports"),
    ("Marathon runners train for months to build the endurance needed for 26.2 miles.", "sports"),
    ("The Olympic Games bring together athletes from nearly every country.", "sports"),
    ("Classical composers like Beethoven shaped the foundations of orchestral music.", "music"),
    ("Jazz emerged in the early twentieth century blending blues and ragtime.", "music"),
    ("Oil painting techniques allow artists to build up layers of rich color.", "arts"),
]

RAW_QUERIES: list[tuple[str, int, str]] = [
    ("fast fox leaping over a dog", 0, "animals"),
    ("animal known for having a great memory", 1, "animals"),
    ("how do dolphins find prey without seeing it", 2, "animals"),
    ("how do plants make energy from light", 3, "biology"),
    ("genetic instructions inside living cells", 4, "biology"),
    ("why can't light escape some space objects", 5, "physics"),
    ("famous river city that is the capital of france", 6, "geography"),
    ("ancient wall that spans thousands of miles in china", 7, "geography"),
    ("tallest mountain in the world", 8, "geography"),
    ("best programming language for analytics", 9, "technology"),
    ("artificial intelligence brain-like networks", 10, "technology"),
    ("decentralized ledger technology for secure records", 11, "technology"),
    ("heart health and working out", 12, "health"),
    ("eating vegetables for long term wellness", 13, "health"),
    ("why is sleep important for memory", 14, "health"),
    ("temperature at which water turns to steam", 15, "physics"),
    ("basic building blocks of matter", 16, "physics"),
    ("is light a wave or a particle", 17, "physics"),
    ("famous landmark built in nineteenth century france", 18, "history"),
    ("when did the roman empire collapse", 19, "history"),
    ("invention that spread information across europe", 20, "history"),
    ("japanese dish with raw fish and rice", 21, "food"),
    ("how coffee gets its flavor", 22, "food"),
    ("fermented foods like kimchi and yogurt", 23, "food"),
    ("world's most popular sport", 24, "sports"),
    ("training for a 26 mile race", 25, "sports"),
    ("international athletic competition every four years", 26, "sports"),
    ("beethoven and orchestral music history", 27, "music"),
    ("music genre blending blues and ragtime", 28, "music"),
    ("layering color technique in painting", 29, "arts"),
]

BUILTIN_DOCUMENTS: list[DocumentItem] = [
    DocumentItem(
        id=f"doc_{idx+1:02d}",
        text=text,
        topic=topic,
        index=idx,
    )
    for idx, (text, topic) in enumerate(RAW_DOCUMENTS)
]

BUILTIN_QUERIES: list[QueryItem] = [
    QueryItem(
        id=f"retrieval_query_{idx+1:02d}",
        text=query_text,
        expected_doc_id=f"doc_{expected_idx+1:02d}",
        expected_index=expected_idx,
        topic=topic,
    )
    for idx, (query_text, expected_idx, topic) in enumerate(RAW_QUERIES)
]


def get_default_corpus() -> tuple[list[DocumentItem], list[QueryItem]]:
    """Return default built-in documents and queries."""
    return list(BUILTIN_DOCUMENTS), list(BUILTIN_QUERIES)
