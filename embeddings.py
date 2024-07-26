from scipy import spatial


def distances_from_embeddings(
    query_embedding: list[float],
    embeddings: list[list[float]],
    distance_metric="cosine",
) -> list[list]:
    distance_metrics = {
        "cosine": spatial.distance.cosine,
        "L1": spatial.distance.cityblock,
        "L2": spatial.distance.euclidean,
        "Linf": spatial.distance.chebyshev,
    }
    distances = [
        distance_metrics[distance_metric](query_embedding, embedding)
        for embedding in embeddings
    ]
    return distances
