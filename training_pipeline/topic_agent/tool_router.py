"""Qwen app routing, lazy schemas and real tool execution using bounded history.

Routing and open_apps are INTERNAL operations. Only env.call() actions enter
the official submission/state. The same Qwen adapter learns all stages.
"""
import copy
import json
from agent_core import (SYSTEM,add_observation,canonical_call,dumps,initial_messages,
                        installed_tool_names,parse_action,prompt_ids)

ROUTER_SYSTEM="""MODE=ROUTER
Bạn chọn nhóm công cụ cho trợ lý LifeMate. Đọc HISTORY và FINAL_REQUEST.
MENU liệt kê toàn bộ nhóm app có sẵn và các khả năng ngắn gọn.
Chọn một vài nhóm cần cho hành động hữu ích đầu tiên, gồm nhóm tra cứu phụ trợ nếu cần.
Có thể mở thêm nhóm trong giai đoạn thực hiện. Nhóm sys luôn có sẵn.
Chỉ trả JSON {"apps":["app_name",...]}; không arguments, không markdown, không giải thích.
Nếu không cần nhóm app nào, trả {"apps":[]}. Không bịa tên ngoài MENU.
HISTORY là các việc đã diễn ra; chỉ xử lý FINAL_REQUEST, không làm lại HISTORY.
"""
EXECUTOR_SYSTEM="""MODE=EXECUTOR
"""+SYSTEM+"""
Một số nhóm app đã mở schema trong TOOLS. AVAILABLE_APPS là các nhóm có thể mở thêm.
Nếu cần công cụ thuộc nhóm chưa mở, trả {"open_apps":["app_name",...]} trước; đây là yêu cầu nội bộ, chưa thực thi tool.
Đọc schema được bổ sung rồi mới sinh tool call thật. Không bịa arguments hoặc ID từ tên menu.
"""


def clean_context(item,tools,cfg):
    source=initial_messages(item,tools,cfg['global_meta_tools'])
    h,request=source[1]['content'].split('\nFINAL_REQUEST=',1)
    return json.loads(h.removeprefix('HISTORY=')),request,json.loads(source[0]['content'].split('\nCONTEXT=',1)[1])


def available_apps(names):return sorted({n.split('.',1)[0] for n in names if not n.startswith('sys.')})


def short_menu(names,tools,description_chars):
    result=[]
    for app in available_apps(names):
        functions=[]
        for name in names:
            if name.split('.',1)[0]!=app:continue
            leaf=name.split('.',1)[-1]
            functions.append({'name':leaf,'about':str(tools[name].get('description',''))[:description_chars]}
                             if description_chars>0 else leaf)
        result.append({'app':app,'tools':functions})
    return result


def router_messages(ctx,tools,cfg,tokenizer):
    if cfg.get("menu_agent"):
        from menu_agent import router_prompt
        return router_prompt(ctx,tools,cfg,tokenizer)
    system=ROUTER_SYSTEM
    if cfg.get('thinking_mode'):
        from thinking_policy import ACTION_RULE
        system+=ACTION_RULE
    for chars in (int(cfg['router_description_chars']),32,0,-1):
        menu=short_menu(ctx['names'],tools,chars) if chars>=0 else [{'app':a} for a in available_apps(ctx['names'])]
        messages=[{'role':'system','content':system+'\nMENU='+dumps(menu)+'\nCONTEXT='+dumps(ctx['public_context'])},
                  {'role':'user','content':'HISTORY='+dumps(ctx['history'])+'\nFINAL_REQUEST='+ctx['request']}]
        if len(prompt_ids(tokenizer,messages))<=cfg['router_prompt_tokens']:break
    return messages


def parse_apps(raw,allowed,key='apps'):
    obj=json.loads(raw)
    if not isinstance(obj,dict) or set(obj)!={key} or not isinstance(obj[key],list):raise ValueError('Return only '+key+' array')
    if any(not isinstance(a,str) or a not in allowed for a in obj[key]):raise ValueError('Unknown app name')
    return sorted(set(obj[key]))


def opened_names(ctx,opened,cfg):
    return [n for n in ctx['names'] if n.split('.',1)[0] in opened or n.startswith('sys.') or n in cfg['global_meta_tools']]


def executor_system(ctx,opened,tools,cfg):
    from context_rerank import spec_for
    specs=[spec_for(tools[n],cfg,True) for n in opened_names(ctx,set(opened),cfg)]
    system=EXECUTOR_SYSTEM
    if cfg.get('thinking_mode'):
        from thinking_policy import ACTION_RULE
        system+=ACTION_RULE
    return system+'\nAVAILABLE_APPS='+dumps(short_menu(ctx['names'],tools,0))+'\nOPEN_APPS='+dumps(sorted(opened))+\
           '\nTOOLS='+dumps(specs)+'\nCONTEXT='+dumps(ctx['public_context'])


def executor_messages(ctx,opened,tools,cfg):
    return [{'role':'system','content':executor_system(ctx,opened,tools,cfg)},
            {'role':'user','content':'HISTORY='+dumps(ctx['history'])+'\nFINAL_REQUEST='+ctx['request']}]


def open_apps(messages,ctx,opened,requested,tools,cfg):
    added=sorted(set(requested)-set(opened));opened.update(requested)
    messages[0]['content']=executor_system(ctx,opened,tools,cfg)
    messages.append({'role':'assistant','content':dumps({'open_apps':requested})})
    messages.append({'role':'user','content':'APP_SCHEMAS_OPENED='+dumps(added)+'. Tiếp tục FINAL_REQUEST bằng schema hiện có.'})
    return added


def compile_routed_context(item,tools,ranker,tokenizer,cfg):
    from history_memory import MemoryCache,memory_key,retrieve_history
    history,request,public_context=clean_context(item,tools,cfg)
    cache=MemoryCache(__import__('pathlib').Path(cfg['output_dir'])/'history_memory.sqlite')
    try:bundle=cache.get(memory_key(history,cfg))
    finally:cache.close()
    if bundle is None or not bundle.get('compression_complete'):
        raise ValueError('Prepare and cache history segmentation/compression before reranking')
    kept,ha=retrieve_history(history,bundle,request,ranker,tokenizer,cfg)
    names=installed_tool_names(item,tools,cfg['global_meta_tools'])
    ctx={'history':kept,'request':request,'public_context':public_context,'names':names}
    messages=router_messages(ctx,tools,cfg,tokenizer)
    # The input bundle is label-free. Gold routing exists only as a later SFT
    # target, never as an initial app selection in validation/test/public.
    item['_router_context']=ctx
    from context_rerank import history_blocks,context_budget
    from context_window import fit_prompt
    _,ids,trim=fit_prompt(tokenizer,messages,cfg)
    old=len(history_blocks(history));selected=ha['selected_block_ids'];recent=ha['recent_block_ids']
    audit={'id':str(item['id']),'available_tools':names,'selected_tools':[],
        'visible_tools':[],'full_schema_tools':[],'compact_fallback_tools':[],
        'tool_schema_policy':'qwen_app_router','detailed_tools_in_prompt':[],
        'descriptions_removed_for_budget':[],'history_blocks_before':old,
        'history_blocks_selected':selected,'history_blocks_kept':selected,
        'retrieved_history_blocks':sorted(set(selected)-set(recent)),'recent_history_blocks':recent,
        'old_history_removed_for_budget':[],'history_not_selected_by_topk':sorted(set(range(old))-set(selected)),
        'context_trim':trim,'context_budget':context_budget(cfg),'prompt_target_tokens_diagnostic':cfg['router_prompt_tokens'],
        'over_context_budget':False,'original_prompt_tokens':len(prompt_ids(tokenizer,initial_messages(item,tools,cfg['global_meta_tools']))),
        'compiled_prompt_tokens':len(ids),'history_retrieval':ha,'segmentation':bundle.get('segmentation',{}),
        'note':'No execution schema is selected until Qwen predicts apps; all apps appear in the short menu.'}
    return messages,audit


def training_apps(call,ctx,tools,cfg):
    from context_rerank import dependency_closure
    selected=dependency_closure({call['name']},ctx['names'],tools,cfg)
    return available_apps(selected)


def make_router_sft_rows(item,tools,calls,observations,tokenizer,cfg):
    """Teacher-force previous internal decisions, as with previous gold calls.

    Retrieval/segmentation were already committed without labels. Labels create
    router/open-app/action TARGETS; inference always uses Qwen's predictions.
    """
    if len(calls)!=len(observations):raise ValueError('Every gold call requires one replayed observation')
    ctx=item['_router_context'];step=0
    initial=training_apps(calls[0],ctx,tools,cfg) if calls else []
    yield {'id':str(item['id']),'step':step,'task':'router',
           'messages':router_messages(ctx,tools,cfg,tokenizer),'target':dumps({'apps':initial})};step+=1
    opened=set(initial);messages=executor_messages(ctx,opened,tools,cfg)
    for call,result in zip(calls,observations):
        missing=set(training_apps(call,ctx,tools,cfg))-opened
        if missing:
            requested=sorted(missing)
            yield {'id':str(item['id']),'step':step,'task':'open_apps','messages':copy.deepcopy(messages),
                   'target':dumps({'open_apps':requested})};step+=1
            open_apps(messages,ctx,opened,requested,tools,cfg)
        yield {'id':str(item['id']),'step':step,'task':'tool_call','messages':copy.deepcopy(messages),'target':dumps(call)};step+=1
        add_observation(messages,call,result)
    yield {'id':str(item['id']),'step':step,'task':'done','messages':copy.deepcopy(messages),'target':dumps({'done':True})}


def routed_rollout(item,tools,env,generate,cfg):
    if cfg.get("menu_agent"):
        from menu_agent import execute
        return execute(item,tools,env,generate,cfg)
    ctx=item['_router_context'];allowed=set(ctx['names']);apps=available_apps(ctx['names'])
    trace=[];invalid=0;error=None;trims=[];route_error=None;initial=None;qwen_calls=0
    thinking_events=[]
    from read_guard import ReadRepeatGuard
    guard=ReadRepeatGuard() if cfg.get("duplicate_read_guard",False) else None
    def next_action(messages):
        try:
            return generate(messages)
        finally:
            event=getattr(generate,'last_thinking',None)
            if event is not None:
                thinking_events.append(event)
    routing=copy.deepcopy(item['_agent_messages'])
    for attempt in range(int(cfg.get('router_max_repairs',1))+1):
        try:
            qwen_calls+=1
            raw=next_action(routing);initial=parse_apps(raw,apps)
            trace.append({'phase':'router','attempt':attempt,'raw':raw,'apps':initial});break
        except (ValueError,TypeError,KeyError) as exc:
            route_error=str(exc)
            routing.append({'role':'user','content':'ROUTER_FORMAT_ERROR='+route_error+'. Trả đúng JSON apps, chỉ dùng tên trong MENU.'})
        except RuntimeError as exc:
            error='router generation: '+str(exc);break
    fallback=initial is None
    if fallback:initial=list(apps)
    opened=set(initial);messages=executor_messages(ctx,opened,tools,cfg)
    if error is None:
        for step in range(int(cfg['max_agent_steps'])):
            try:
                qwen_calls+=1
                raw=next_action(messages);trim=getattr(generate,'last_context_trim',None)
                if trim and trim.get('truncated'):trims.append({'step':step,**trim})
            except RuntimeError as exc:error='generation: '+str(exc);break
            except ValueError as exc:
                if not cfg.get('thinking_mode'):
                    error='generation: '+str(exc);break
                invalid+=1
                trace.append({'phase':'repair','step':step,'unexecuted_error':str(exc)})
                if invalid>cfg['max_repairs']:
                    error='Exceeded Thinking generation repair budget';break
                messages.append({'role':'user','content':'Output chưa hoàn chỉnh. Kết thúc suy luận sớm và trả đúng một JSON action.'})
                continue
            try:
                obj=json.loads(raw)
                if isinstance(obj,dict) and set(obj)=={'open_apps'}:
                    requested=parse_apps(raw,apps,'open_apps')
                    if not set(requested)-opened:raise ValueError('Requested apps are already open; execute the next action')
                    added=open_apps(messages,ctx,opened,requested,tools,cfg)
                    trace.append({'phase':'open_apps','step':step,'apps':added});continue
                action=parse_action(raw)
                if action.get('done') is True:trace.append({'phase':'done','step':step,'raw':raw});break
                if action['name'] not in allowed:raise ValueError('Tool is not installed for this sample')
                app=action['name'].split('.',1)[0]
                if action['name'] not in opened_names(ctx,opened,cfg):
                    # Load a missing schema, but never execute an unseen-schema
                    # call. The model must regenerate it after reading schema.
                    added=open_apps(messages,ctx,opened,[app],tools,cfg)
                    trace.append({'phase':'auto_open','step':step,'apps':added,'unexecuted_call':action});continue
                validation=env.validate(action['name'],action['arguments'])
                if not validation.get('ok'):raise ValueError('Schema: '+dumps(validation))
            except (ValueError,TypeError,KeyError) as exc:
                invalid+=1;trace.append({'phase':'repair','step':step,'raw':raw,'unexecuted_error':str(exc)})
                messages.extend([{'role':'assistant','content':raw},{'role':'user','content':'FORMAT_OR_SCHEMA_ERROR='+str(exc)+'. Chưa thực thi. Sửa đúng một JSON.'}])
                if invalid>cfg['max_repairs']:error='Exceeded format/schema repair budget';break
                continue
            kind=tools[action['name']].get('kind','unknown')
            if guard is not None:
                cached,previous=guard.lookup(action,kind)
                if cached:
                    trace.append({'phase':'cached_read','step':step,'call':action,'previous_result':previous,'executed':False})
                    messages.extend([{'role':'assistant','content':raw}, {'role':'user','content':
                        'DUPLICATE_READ_NOT_EXECUTED: Cùng lệnh đọc và tham số đã có kết quả, chưa có lệnh không-phải-đọc xen giữa. '
                        'Dùng kết quả trước để quyết định bước tiếp theo hoặc kết thúc. PREVIOUS_RESULT='+dumps(previous)}])
                    if guard.blocks>=int(cfg.get('max_duplicate_read_blocks',3)):
                        error='Repeated identical read loop stopped without re-executing the calls';break
                    continue
            result=env.call(action['name'],action['arguments'])
            if guard is not None:guard.record(action,result,kind)
            trace.append({'phase':'tool','step':step,'call':action,'result':result});add_observation(messages,action,result)
        else:error='Reached max_agent_steps without done'
    return {'id':str(item['id']),'tool_calls':env.get_tool_calls(),'state_hash':env.get_state_hash(),
        'trace':trace,'invalid_model_actions':invalid,'error':error,'context_trims':trims,
        **({'thinking':thinking_events} if cfg.get('thinking_mode') else {}),
        'router':{'initial_apps':initial,'final_apps':sorted(opened),'fallback_all_apps':fallback,
                  'format_error':route_error,'qwen_calls':qwen_calls}}
