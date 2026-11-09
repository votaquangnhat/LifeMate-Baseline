"""Official icefall greedy/modified-beam decoding for the stateless Gipformer predictor."""
import torch
from text_processing import spoken_text

def greedy_ids(asr, encoded, lengths, max_symbols=1):
    from beam_search import greedy_search_batch
    return greedy_search_batch(model=asr, encoder_out=encoded, encoder_out_lens=lengths)

def beam_ids(asr, encoded, length, tokenizer, phrases=(), cfg=None):
    from beam_search import modified_beam_search
    from icefall import ContextGraph
    cfg = cfg or {}
    graph = None
    patterns, texts, seen = [], [], set()
    for phrase in list(phrases)[:cfg.get('max_context_phrases',128)]:
        text = spoken_text(phrase,cfg)
        ids, missing = tokenizer.encode(text)
        if ids and not missing and len(ids)<=32 and tuple(ids) not in seen:
            patterns.append(ids); texts.append(text); seen.add(tuple(ids))
    if patterns and cfg.get('context_weight',.6)>0:
        graph = ContextGraph(context_score=float(cfg.get('context_weight',.6)))
        graph.build(token_ids=patterns, phrases=texts)
    result = modified_beam_search(model=asr, encoder_out=encoded,
        encoder_out_lens=torch.tensor([length],device=encoded.device),
        context_graph=graph, beam=int(cfg.get('beam_width',4)))
    return result[0]

