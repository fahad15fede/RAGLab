
import math


def reciprocal_rank_fusion(ranked_lists, k=60):
    """
    Combine ranked lists using Reciprocal Rank Fusion.

    ranked_lists: list of lists containing chunk IDs.
    Returns: chunk IDs sorted by fused score.
    """
    scores = {}

    for ranked_ids in ranked_lists:
        for rank, chunk_id in enumerate(ranked_ids, start=1):
            scores[chunk_id] = (
                scores.get(chunk_id, 0.0)
                + 1.0 / (k + rank)
            )

    return sorted(
        scores,
        key=lambda chunk_id: (-scores[chunk_id], chunk_id)
    )


def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    if len(a) != len(b):
        raise ValueError("Vector dimensions must match")

    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)


def mmr_select(
    query_vector,
    candidates,
    k=5,
    lambda_mult=0.5
):
    """
    candidates: list of dictionaries:
        {"id": str, "embedding": list[float]}

    Returns selected candidate IDs in MMR order.
    """
    if not 0 <= lambda_mult <= 1:
        raise ValueError("lambda_mult must be between 0 and 1")

    if k <= 0:
        return []

    remaining = list(candidates)
    selected = []

    while remaining and len(selected) < k:
        best = None
        best_score = float("-inf")

        for candidate in remaining:
            relevance = cosine_similarity(
                query_vector,
                candidate["embedding"]
            )

            redundancy = max(
                (
                    cosine_similarity(
                        candidate["embedding"],
                        item["embedding"]
                    )
                    for item in selected
                ),
                default=0.0
            )

            score = (
                lambda_mult * relevance
                - (1 - lambda_mult) * redundancy
            )

            if score > best_score:
                best_score = score
                best = candidate

        selected.append(best)
        remaining.remove(best)

    return [item["id"] for item in selected]
