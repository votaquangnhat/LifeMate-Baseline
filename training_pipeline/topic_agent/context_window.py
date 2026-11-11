
"""Bound every model prompt; trim context without dropping a training sample."""
import copy
import json
from agent_core import dumps, prompt_ids


def clip_text(tokenizer, text, limit):
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= limit:
        return text
    head = max(1,limit//3)
    tail = max(0,limit-head)
    if hasattr(tokenizer,"decode"):
        a=tokenizer.decode(ids[:head],skip_special_tokens=False,clean_up_tokenization_spaces=False)
        b=tokenizer.decode(ids[-tail:],skip_special_tokens=False,clean_up_tokenization_spaces=False) if tail else ""
    else:
        a,b=text[:head],text[-tail:] if tail else ""
    return a+" [...context shortened...] "+b


def compact_result(value, tokenizer, text_limit, item_limit):
    if isinstance(value,dict):
        exact={"id","phone","phone_number","date","time","start","end","amount"}
        return {k:(copy.deepcopy(v) if (k in exact or k.endswith("_id")) and isinstance(v,(str,int,float,bool))
                   else compact_result(v,tokenizer,text_limit,item_limit)) for k,v in value.items()}
    if isinstance(value,list):
        selected=value if len(value)<=item_limit else value[:max(1,item_limit//2)]+value[-(item_limit//2):] if item_limit>1 else value[:1]
        return [compact_result(v,tokenizer,text_limit,item_limit) for v in selected]
    if isinstance(value,str):
        return clip_text(tokenizer,value,text_limit)
    return value


def fit_prompt(tokenizer, messages, cfg, prompt_limit=None):
    """Same context policy for initial compilation, every SFT step and inference.

    Returns valid chat messages plus exact bounded token IDs. Only the extreme
    final fallback clips encoded prompt IDs, preserving its head and latest tail.
    Gold target tokens are never passed to this function.
    """
    if cfg.get("agent_variant") == "enhanced" or cfg.get("protected_context"):
        from context_preserving import fit_prompt_preserving
        return fit_prompt_preserving(tokenizer,messages,cfg,prompt_limit)
    capacity=int(cfg.get("max_length",cfg.get("max_length_cap",8192)))
    reserve=max(int(cfg.get("requested_max_new_tokens",512)),int(cfg.get("max_new_tokens_cap",512)))
    margin=int(cfg.get("length_safety_margin",128))
    limit=int(prompt_limit if prompt_limit is not None else capacity-reserve-margin)
    if cfg.get('history_router') and messages:
        system=str(messages[0].get('content',''))
        if system.startswith('MODE=ROUTER'):limit=min(limit,int(cfg['router_prompt_tokens']))
        elif system.startswith('MODE=EXECUTOR'):limit=min(limit,int(cfg['executor_prompt_tokens']))
    if limit<=0:
        raise ValueError("Invalid configuration: no capacity remains for a prompt")
    m=copy.deepcopy(messages)
    ids=prompt_ids(tokenizer,m);before=len(ids)
    audit={"before_tokens":before,"after_tokens":before,"limit":limit,"truncated":False,
           "history_blocks_removed":0,"tool_results_compacted":0,"old_messages_removed":0,
           "schema_names_only":False,"descriptions_removed":False,"system_content_shortened":False,"emergency_token_clip":False}
    def over():
        nonlocal ids
        ids=prompt_ids(tokenizer,m)
        return len(ids)>limit
    if over():
        if m and m[0].get("role")=="system":
            content=m[0].get("content","")
            if "\nTOOLS=" in content and "\nCONTEXT=" in content:
                prefix,rest=content.split("\nTOOLS=",1);specs,context=rest.split("\nCONTEXT=",1)
                try:
                    from context_rerank import compact_schema
                    compact=dumps([compact_schema(s,0) for s in json.loads(specs)])
                    m[0]["content"]=prefix+"\nTOOLS="+compact+"\nCONTEXT="+context
                    audit["descriptions_removed"]=True
                except (ValueError,TypeError):pass
    if over():
        # Initial history is JSON data inside the first user message.
        for msg in m:
            content=msg.get("content","")
            if msg.get("role")=="user" and isinstance(content,str) and content.startswith("HISTORY=") and "\nFINAL_REQUEST=" in content:
                raw,request=content.split("\nFINAL_REQUEST=",1)
                try:history=json.loads(raw.removeprefix("HISTORY="))
                except ValueError:break
                blocks=[]
                for turn in history:
                    if turn.get("role")=="user" or not blocks:blocks.append([])
                    blocks[-1].append(turn)
                while blocks and over():
                    blocks.pop(0);audit["history_blocks_removed"]+=1
                    msg["content"]="HISTORY="+dumps([t for b in blocks for t in b])+"\nFINAL_REQUEST="+request
                break
    # Keep latest observations longer, then compact older result payloads.
    for text_limit,item_limit in ((512,16),(128,4),(32,1)):
        if not over():break
        for msg in m:
            if not over():break
            if msg.get("role")=="tool" and isinstance(msg.get("content"),str):
                original=msg["content"]
                try:msg["content"]=dumps(compact_result(json.loads(original),tokenizer,text_limit,item_limit))
                except ValueError:msg["content"]=clip_text(tokenizer,original,text_limit)
                audit["tool_results_compacted"]+=int(msg["content"]!=original)
    # Drop oldest completed observation/call pairs only when still necessary.
    while over() and len(m)>4:
        end=4 if m[2].get("role")=="assistant" and m[3].get("role")=="tool" else 3
        audit["old_messages_removed"]+=end-2
        del m[2:end]
    if over() and m and m[0].get("role")=="system":
        content=m[0].get("content","")
        if "\nTOOLS=" in content and "\nCONTEXT=" in content:
            prefix,rest=content.split("\nTOOLS=",1);tools,context=rest.split("\nCONTEXT=",1)
            try:
                names=[{"name":t["name"]} for t in json.loads(tools)]
                m[0]["content"]=prefix+"\nTOOLS="+dumps(names)+"\nCONTEXT="+context
                audit["schema_names_only"]=True
            except (ValueError,KeyError,TypeError):pass
    # Rare oversized request/system/observation: retain head and tail, not rows.
    for allowance in (max(64,limit//2),max(32,limit//4),32):
        if not over():break
        for msg in sorted(m,key=lambda v:len(v.get("content","")),reverse=True):
            if not over():break
            content=msg.get("content","")
            if isinstance(content,str):
                msg["content"]=clip_text(tokenizer,content,allowance)
                if msg.get("role")=="system" and msg["content"]!=content:audit["system_content_shortened"]=True
    ids=prompt_ids(tokenizer,m)
    if len(ids)>limit:
        head=min(256,limit//4);tail=limit-head
        ids=ids[:head]+ids[-tail:] if head else ids[-tail:]
        audit["emergency_token_clip"]=True
    audit.update(after_tokens=len(ids),truncated=before>len(ids))
    return m,ids,audit
