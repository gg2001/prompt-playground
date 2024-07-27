from scipy import spatial
from embeddings import get_embeddings
from typing import Callable

Distance = Callable[[list[float], list[float]], float]

DISTANCE_METRIC = "cosine"
TOP_N = 100


def query_embeddings(
    query: str,
    embeddings: list[list[float]],
    top_n: int = TOP_N,
    distance_metric=DISTANCE_METRIC,
) -> list[tuple[int, float]]:
    query_embedding = get_embeddings([query])
    return retrieve_from_embeddings(
        query_embedding[0], embeddings, top_n, distance_metric
    )


def retrieve_from_embeddings(
    query_embedding: list[float],
    embeddings: list[list[float]],
    top_n: int = TOP_N,
    distance_metric=DISTANCE_METRIC,
) -> list[tuple[int, float]]:
    distances = distances_from_embeddings(query_embedding, embeddings, distance_metric)
    indexed_distances = list(enumerate(distances))
    indexed_distances.sort(key=lambda x: x[1], reverse=True)
    return indexed_distances[:top_n]


def distances_from_embeddings(
    query_embedding: list[float],
    embeddings: list[list[float]],
    distance_metric=DISTANCE_METRIC,
) -> list[float]:
    distance_metrics: dict[str, Distance] = {
        "cosine": lambda x, y: spatial.distance.cosine(x, y).item(),
        "L1": spatial.distance.cityblock,
        "L2": spatial.distance.euclidean,
        "Linf": spatial.distance.chebyshev,
    }
    distances = [
        distance_metrics[distance_metric](query_embedding, embedding)
        for embedding in embeddings
    ]
    return distances
