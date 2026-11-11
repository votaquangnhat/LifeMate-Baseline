"""Per-sample topical memory, CPU BM25 and bounded semantic reranking.

No search across users/samples. Raw turns are the values; LLMLingua-2 text is
an extra retrieval key, never the text from which the agent copies arguments.
"""
import collections
import copy
import hashlib
import json
import math
import re
import sqlite3
import unicodedata
from pathlib import Path
from agent_core import dumps

HISTORY_VERSION='secom-inspired-bm25-bge-v1'
HISTORY_KEYS=('history_token_budget','history_bm25_candidates','history_bm25_k1','history_bm25_b',
    'history_diversity','keep_recent_history','segment_window_tokens','segment_window_blocks',
    'segment_max_tokens','segment_short_history_tokens','segment_max_new_tokens',
    'history_compression_rate','history_compressor_id','history_compressor_revision')


def history_signature(cfg):
    return {'version':HISTORY_VERSION,**{k:cfg[k] for k in HISTORY_KEYS},
            'segmenter_model':cfg.get('segmenter_model_id',cfg['model_id']),
            'segmenter_revision':cfg.get('segmenter_model_revision',cfg['model_revision'])}


def words(text):
    return re.findall(r"[^\W_]+(?:[_'-][^\W_]+)*",unicodedata.normalize('NFC',text).lower(),re.UNICODE)


def block_text(block):
    lines=[]
    for turn in block:
        content={k:v for k,v in turn.items() if k!='role'}
        lines.append(str(turn.get('role',''))+': '+(content.get('text','') if set(content)=={'text'} else dumps(content)))
    return '\n'.join(lines)


def memory_key(history,cfg):
    return hashlib.sha256(dumps([history_signature(cfg),history]).encode()).hexdigest()


class MemoryCache:
    def __init__(self,path):
        Path(path).parent.mkdir(parents=True,exist_ok=True)
        self.db=sqlite3.connect(str(path))
        self.db.execute('CREATE TABLE IF NOT EXISTS memory (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
    def get(self,key):
        row=self.db.execute('SELECT value FROM memory WHERE key=?',(key,)).fetchone()
        return json.loads(row[0]) if row else None
    def put(self,key,value):
        self.db.execute('INSERT OR REPLACE INTO memory VALUES (?,?)',(key,dumps(value)));self.db.commit()
    def close(self):self.db.close()


def bm25_scores(query,documents,k1=1.5,b=.75):
    """Positive-IDF BM25; Unicode syllables/words and numbers, CPU only."""
    corpus=[words(d) for d in documents]
    n=len(corpus)
    if not n:return []
    avg=sum(map(len,corpus))/n or 1.
    counts=[collections.Counter(d) for d in corpus];df=collections.Counter()
    for c in counts:df.update(c.keys())
    terms=set(words(query));scores=[]
    for tokens,c in zip(corpus,counts):
        score=0.
        for term in terms:
            tf=c[term]
            if tf:
                idf=math.log(1+(n-df[term]+.5)/(df[term]+.5))
                score+=idf*tf*(k1+1)/(tf+k1*(1-b+b*len(tokens)/avg))
        scores.append(score)
    return scores


def bm25_candidates(query,segments,cfg):
    """Fuse raw and compressed indexes; return a bounded candidate pool."""
    if not segments:return [],{}
    raw=bm25_scores(query,[s['raw_text'] for s in segments],cfg['history_bm25_k1'],cfg['history_bm25_b'])
    compressed=bm25_scores(query,[s.get('index_text',s['raw_text']) for s in segments],cfg['history_bm25_k1'],cfg['history_bm25_b'])
    rr=[0.]*len(segments)
    for scores in (raw,compressed):
        positive=[i for i in range(len(scores)) if scores[i]>0]
        for rank,i in enumerate(sorted(positive,key=lambda i:(-scores[i],-i))):rr[i]+=1/(60+rank+1)
    order=sorted(range(len(segments)),key=lambda i:(-rr[i],-i))[:int(cfg['history_bm25_candidates'])]
    return order,{'raw_bm25':raw,'compressed_bm25':compressed,'rrf':rr}


def parse_ranges(raw,start,end):
    obj=json.loads(raw)
    if not isinstance(obj,dict) or set(obj)!={'segments'} or not isinstance(obj['segments'],list):raise ValueError('Expected segments array')
    ranges=obj['segments'];next_start=start
    for pair in ranges:
        if not isinstance(pair,list) or len(pair)!=2 or any(type(x) is not int for x in pair):raise ValueError('Bad segment range')
        a,b=pair
        if a!=next_start or b<a or b>end:raise ValueError('Segments must cover each history block once in order')
        next_start=b+1
    if next_start!=end+1:raise ValueError('Missing history blocks')
    return ranges


def segmentation_windows(blocks,tokenizer,cfg):
    """Bound the segmenter's input; original history remains unmodified."""
    from context_window import clip_text
    windows=[];current=[]
    for i,block in enumerate(blocks):
        text=clip_text(tokenizer,block_text(block),max(64,int(cfg['segment_window_tokens'])//2))
        row={'block':i,'text':text}
        if current and (len(current)>=cfg['segment_window_blocks'] or len(tokenizer.encode(dumps(current+[row]),add_special_tokens=False))>cfg['segment_window_tokens']):
            windows.append(current);current=[]
        current.append(row)
    if current:windows.append(current)
    return windows


def bounded_segments(blocks,ranges,tokenizer,cfg):
    segments=[]
    for a,b in ranges:
        ids=[]
        for i in range(a,b+1):
            candidate=ids+[i]
            text='\n'.join(block_text(blocks[j]) for j in candidate)
            if ids and len(tokenizer.encode(text,add_special_tokens=False))>cfg['segment_max_tokens']:
                segments.append(ids);ids=[]
            ids.append(i)
        if ids:segments.append(ids)
    return [{'block_ids':ids,'raw_text':'\n'.join(block_text(blocks[i]) for i in ids)} for ids in segments]


def bound_history(blocks,selected,tokenizer,budget):
    from context_window import clip_text,compact_result
    selected=sorted(set(selected));history=[copy.deepcopy(t) for i in selected for t in blocks[i]]
    def length():return len(tokenizer.encode(dumps(history),add_special_tokens=False))
    shortened=False
    while len(selected)>1 and length()>budget:
        selected.pop(0);history=[copy.deepcopy(t) for i in selected for t in blocks[i]];shortened=True
    if length()>budget:
        history=compact_result(history,tokenizer,max(32,budget//8),4);shortened=True
    if length()>budget:
        text='\n'.join(block_text(blocks[i]) for i in selected)
        history=[{'role':'user','text':clip_text(tokenizer,text,max(1,budget//2))}]
    if len(tokenizer.encode(dumps(history),add_special_tokens=False))>budget:
        history=[{'role':'user','text':clip_text(tokenizer,dumps(history),max(1,budget//4))}]
    allowance=max(1,budget//8)
    while len(tokenizer.encode(dumps(history),add_special_tokens=False))>budget:
        if allowance==1:history=[];selected=[];break
        history=[{'role':'user','text':clip_text(tokenizer,dumps(history),allowance)}]
        allowance=max(1,allowance//2);shortened=True
    return history,selected,shortened


def retrieve_history(history,bundle,request,ranker,tokenizer,cfg):
    if cfg.get("agent_variant") == "enhanced" or cfg.get("protected_context"):
        from context_preserving import retrieve_history_preserving
        return retrieve_history_preserving(history,bundle,request,ranker,tokenizer,cfg)
    from context_rerank import history_blocks
    blocks=history_blocks(history);budget=int(cfg['history_token_budget'])
    recent=list(range(max(0,len(blocks)-int(cfg['keep_recent_history'])),len(blocks)))
    query=request+'\nRecent context:\n'+'\n'.join(block_text(blocks[i]) for i in recent)
    full_tokens=len(tokenizer.encode(dumps(history),add_special_tokens=False))
    if full_tokens<=budget:
        return copy.deepcopy(history),{'full_history':True,'selected_block_ids':list(range(len(blocks))),
            'recent_block_ids':recent,'candidate_segments':[],'rerank_scores':{},'history_tokens':full_tokens,
            'source_history_tokens':full_tokens,'emergency_history_clip':False}
    segments=bundle['segments'];candidates,lexical=bm25_candidates(query,segments,cfg)
    recent_set=set(recent)
    candidates=list(dict.fromkeys(candidates+[i for i,s in enumerate(segments) if recent_set&set(s['block_ids'])]))
    scores=ranker.score(query,[segments[i]['raw_text'] for i in candidates])
    order=sorted(range(len(candidates)),key=lambda j:(-scores[j],-candidates[j]))
    selected=set(recent);selected_segments=[]
    def cost(ids):return len(tokenizer.encode(dumps([t for i in sorted(ids) for t in blocks[i]]),add_special_tokens=False))
    # Greedy rank with redundancy penalty; no fixed number of returned blocks.
    pending=list(order)
    while pending:
        def priority(j):
            ids=set(segments[candidates[j]]['block_ids']);overlap=len(ids&selected)/max(1,len(ids))
            return 1/(1+order.index(j))-cfg['history_diversity']*overlap
        j=max(pending,key=priority);pending.remove(j);i=candidates[j]
        proposed=selected|set(segments[i]['block_ids'])
        if proposed!=selected and cost(proposed)<=budget:selected=proposed;selected_segments.append(i)
    result,kept,shortened=bound_history(blocks,selected,tokenizer,budget)
    return result,{'full_history':False,'selected_block_ids':kept,'recent_block_ids':recent,
        'candidate_segments':candidates,'selected_segments':selected_segments,
        'rerank_scores':{str(i):scores[j] for j,i in enumerate(candidates)},'bm25':lexical,
        'history_tokens':len(tokenizer.encode(dumps(result),add_special_tokens=False)),
        'source_history_tokens':full_tokens,'emergency_history_clip':shortened}
