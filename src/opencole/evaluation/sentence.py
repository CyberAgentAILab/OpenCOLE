from sentence_transformers import SentenceTransformer, util


def compute_sentence_similarity_over_pairs(
    model_name_or_path: str, pairs: list[tuple[str, str]]
) -> float:
    model = SentenceTransformer(model_name_or_path)
    scores = []
    for pair in pairs:
        sim = compute_sentence_similarity(pair[0], pair[1], model)
    scores.append(sim)
    return sum(scores) / len(scores)


def compute_sentence_similarity(
    sentences1: str, sentences2: str, model: SentenceTransformer
) -> float:
    sentences = [sentences1, sentences2]
    emb = model.encode(sentences)
    cos_sim = util.cos_sim(emb[0], emb[1])
    return float(cos_sim)
