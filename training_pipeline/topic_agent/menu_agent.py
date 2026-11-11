"""All-app menu, reversible full-schema loading, and optional bounded recovery.

Uses the existing topical adapter's apps/open_apps/tool-call/done JSON protocol.
No gold labels or simulator state are consulted when deciding what to load.
"""
import copy
import json
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from agent_core import SYSTEM, dumps, parse_action, add_observation, prompt_ids
from thinking_policy import ACTION_RULE
from context_window import fit_prompt
from read_guard import ReadRepeatGuard


def calendar_context(context):
    result=copy.deepcopy(context)
    stamp=context.get('current_time')
    if not isinstance(stamp,str):return result
    try:
        clock=datetime.fromisoformat(stamp.replace('Z','+00:00'))
        if clock.tzinfo is not None and context.get('timezone'):
            clock=clock.astimezone(ZoneInfo(context['timezone']))
    except (ValueError,KeyError):return result
    today=clock.date();monday=today-timedelta(days=today.weekday())
    days=('Thứ Hai','Thứ Ba','Thứ Tư','Thứ Năm','Thứ Sáu','Thứ Bảy','Chủ Nhật')
    result['date_reference']={'source':'context.current_time and timezone',
        'today':today.isoformat(),
        'days':[{'date':(today+timedelta(days=d)).isoformat(),'weekday':days[(today+timedelta(days=d)).weekday()],
                 'offset_from_today':d} for d in range(-1,14)],
        'this_week':[monday.isoformat(),(monday+timedelta(days=6)).isoformat()],
        'next_week':[(monday+timedelta(days=7)).isoformat(),(monday+timedelta(days=13)).isoformat()]}
    return result


def menu(ctx,tools,description_chars=100):
    apps={}
    for name in ctx['names']:
        app=name.split('.',1)[0]
        if app=='sys':continue
        meta=tools[name].get('_app',{})
        row=apps.setdefault(app,{'app':app,'name_vi':meta.get('name_vi',meta.get('name',app)),
            'about':str(meta.get('description',''))[:description_chars], 'tools':[]})
        row['tools'].append(name.split('.',1)[1])
    return [apps[a] for a in sorted(apps)]


def context_view(ctx,cfg):
    return calendar_context(ctx['public_context']) if cfg.get('menu_recovery') else ctx['public_context']


def router_prompt(ctx,tools,cfg,tokenizer):
    instruction=('MODE=ROUTER\nChọn các app cần cho bước hữu ích đầu tiên của FINAL_REQUEST. '
        'MENU liệt kê TẤT CẢ app đã cài, tên tiếng Việt và tên công cụ, không phải schema tham số. '
        'Trả {"apps":["app_id",...]}; có thể mở thêm bất kỳ app trong MENU ở các bước sau. '
        'Không suy ra rằng app không được chọn ban đầu là không khả dụng. Lịch sử chỉ là ngữ cảnh đã diễn ra.\n')+ACTION_RULE
    for chars in (100,48,0):
        messages=[{'role':'system','content':instruction+'\nMENU='+dumps(menu(ctx,tools,chars))+'\nCONTEXT='+dumps(context_view(ctx,cfg))},
                  {'role':'user','content':'HISTORY='+dumps(ctx['history'])+'\nFINAL_REQUEST='+ctx['request']}]
        if len(prompt_ids(tokenizer,messages))<=cfg['router_prompt_tokens']:break
    # Names and tool names are never removed just to meet a soft prompt target.
    return messages


def app_list(raw,allowed,key):
    value=json.loads(raw)
    if not isinstance(value,dict) or set(value)!={key} or not isinstance(value[key],list):
        raise ValueError('Expected only '+key+' array')
    if any(not isinstance(a,str) or a not in allowed for a in value[key]):raise ValueError('App is absent from the installed menu')
    return list(dict.fromkeys(value[key]))


def complete_schema(tool):
    return {k:copy.deepcopy(tool[k]) for k in ('name','kind','description','parameters','returns') if k in tool}


class SchemaWindow:
    def __init__(self,ctx,tools,cfg,tokenizer):
        self.ctx,self.tools,self.cfg,self.tokenizer=ctx,tools,cfg,tokenizer
        self.active=[];self.ever=set();self.evictions=[]
        self.apps={r['app'] for r in menu(ctx,tools)}

    def names(self,active):
        return [n for n in self.ctx['names'] if n.startswith('sys.') or n.split('.',1)[0] in active or n in self.cfg['global_meta_tools']]

    def render(self,messages,active):
        result=copy.deepcopy(messages)
        policy=('MODE=EXECUTOR\n'+SYSTEM+'\n'+ACTION_RULE+
            '\nAVAILABLE_APPS luôn gồm mọi app đã cài. TOOLS chỉ là schema đầy đủ hiện đang nạp. '
            'Muốn dùng app chưa nạp, trả {"open_apps":["app_id"]}; đây là thao tác nội bộ, không gọi ToolEnv. '
            'Có thể mở lại app đã dỡ khỏi bộ nhớ. Không bịa tham số trước khi đọc schema. '
            'Chỉ thực hiện FINAL_REQUEST; không lặp lại việc đã làm trong HISTORY. ')
        if self.cfg.get('menu_recovery'):
            policy+=('Dùng date_reference để kiểm tra ngày tương đối. Trước khi ghi, đối chiếu điều kiện với kết quả đọc thực tế. '
                     'Không đoán ID hay số liệu chưa được cung cấp. Hỏi thông tin bắt buộc còn thiếu nếu không thể tra cứu. ')
        result[0]={'role':'system','content':policy+'\nAVAILABLE_APPS='+dumps(menu(self.ctx,self.tools))+
                   '\nOPEN_APPS='+dumps(active)+'\nTOOLS='+dumps([complete_schema(self.tools[n]) for n in self.names(active)])+
                   '\nCONTEXT='+dumps(context_view(self.ctx,self.cfg))}
        return result

    def fit(self,messages,requested=()):
        requested=list(dict.fromkeys(requested))
        if any(a not in self.apps for a in requested):raise ValueError('Cannot open uninstalled app')
        active=[a for a in self.active if a not in requested]+requested
        preferred=requested[0] if requested else (active[-1] if active else None)
        while True:
            rendered=self.render(messages,active)
            try:
                fitted,_,audit=fit_prompt(self.tokenizer,rendered,self.cfg)
                evicted=[a for a in self.active if a not in active]
                self.evictions.extend(evicted);self.active=active
                return fitted,dict(audit,active_apps=active[:],evicted_apps=evicted,
                                   deferred_apps=[a for a in requested if a not in active])
            except ValueError:
                victims=[a for a in active if a not in requested]
                if not victims:victims=[a for a in reversed(active) if a!=preferred]
                if not victims:raise
                active.remove(victims[0])


def empty_search(result):
    if not isinstance(result,dict) or result.get('ok') is False:return False
    lists=[v for k,v in result.items() if isinstance(v,list) and k not in ('warnings','errors','logs')]
    return bool(lists) and all(not v for v in lists)


def execute(item,tools,env,generate,cfg):
    tokenizer=getattr(generate,'tokenizer',None)
    if tokenizer is None:raise ValueError('Menu agent requires the generator tokenizer for schema budgeting')
    ctx=item['_router_context'];window=SchemaWindow(ctx,tools,cfg,tokenizer)
    trace=[];trims=[];thinking=[];invalid=0;error=None;calls=0;initial=None;route_error=None
    guard=ReadRepeatGuard() if cfg.get('menu_recovery') else None
    reviews=0;empty_hints={};asked=False
    def next_action(messages):
        nonlocal calls
        calls+=1
        try:return generate(messages)
        finally:
            event=getattr(generate,'last_thinking',None)
            if event is not None:thinking.append(copy.deepcopy(event))
    routing=router_prompt(ctx,tools,cfg,tokenizer)
    for attempt in range(int(cfg.get('router_max_repairs',1))+1):
        try:
            raw=next_action(routing);initial=app_list(raw,window.apps,'apps')
            trace.append({'phase':'router','raw':raw,'apps':initial});break
        except (ValueError,TypeError,KeyError) as exc:
            route_error=str(exc);routing.append({'role':'user','content':'ROUTER_FORMAT_ERROR='+route_error+'. Trả đúng JSON apps.'})
        except RuntimeError as exc:
            if 'out of memory' in str(exc).lower():raise
            error='router generation: '+str(exc);break
    fallback=initial is None
    if fallback:initial=[]  # Do not flood the model with all schemas after a bad router answer.
    messages=[{'role':'system','content':''},{'role':'user','content':'HISTORY='+dumps(ctx['history'])+'\nFINAL_REQUEST='+ctx['request']}]
    try:
        if error is None:
            messages,audit=window.fit(messages,initial)
            trace.append({'phase':'schemas','requested_apps':initial,**audit})
            for step in range(int(cfg['max_agent_steps'])):
                messages,audit=window.fit(messages)
                if audit['truncated'] or audit['evicted_apps']:trims.append(dict(audit,step=step))
                window.ever.update(window.active)
                try:raw=next_action(messages)
                except ValueError as exc:
                    invalid+=1;trace.append({'phase':'repair','step':step,'unexecuted_error':str(exc)})
                    if invalid>cfg['max_repairs']:error='Exceeded incomplete-generation repair budget';break
                    messages.append({'role':'user','content':'Chưa có action hoàn chỉnh. Kết thúc suy luận rồi trả một JSON.'});continue
                try:
                    obj=json.loads(raw)
                    if isinstance(obj,dict) and set(obj)=={'open_apps'}:
                        requested=app_list(raw,window.apps,'open_apps')
                        if not requested or set(requested)<=set(window.active):raise ValueError('Requested app schemas are already loaded; choose the next action')
                        requested=[a for a in requested if a not in window.active]+[a for a in requested if a in window.active]
                        messages,audit=window.fit(messages,requested)
                        messages.extend([{'role':'assistant','content':raw},{'role':'user','content':'APP_SCHEMAS_OPENED='+dumps(window.active)+
                            '; DEFERRED_APPS='+dumps(audit['deferred_apps'])+'. Đọc schema trong TOOLS rồi tiếp tục.'}])
                        trace.append({'phase':'open_apps','step':step,**audit});continue
                    action=parse_action(raw)
                    if action.get('done') is True:
                        if cfg.get('menu_recovery') and not asked and reviews==0 and window.apps-window.ever and step+1<cfg['max_agent_steps']:
                            reviews+=1;trace.append({'phase':'completion_review','step':step,'executed':False})
                            messages.extend([{'role':'assistant','content':raw},{'role':'user','content':
                                'Kiểm tra một lần trước khi kết thúc: mọi hành động được yêu cầu đã hoàn tất chưa? '
                                'App chưa mở vẫn khả dụng trong AVAILABLE_APPS; mở schema nếu cần cho việc còn thiếu. '
                                'Không gọi thêm tool chỉ để kiểm tra. Nếu đã xong, ngoài phạm vi, hoặc không còn hành động được yêu cầu, trả {"done":true}.'}]);continue
                        trace.append({'phase':'done','step':step});break
                    if action['name'] not in ctx['names']:raise ValueError('Tool is not installed for this sample')
                    app=action['name'].split('.',1)[0]
                    if action['name'] not in window.names(window.active):
                        messages,audit=window.fit(messages,[app])
                        messages.append({'role':'user','content':'Chưa thực thi lời gọi vì schema chưa nạp. Schema '+app+' đã nạp; đọc rồi sinh lại action.'})
                        trace.append({'phase':'auto_open','step':step,'unexecuted_call':action,**audit});continue
                    validation=env.validate(action['name'],action['arguments'])
                    if not validation.get('ok'):raise ValueError('Schema: '+dumps(validation))
                except (ValueError,TypeError,KeyError) as exc:
                    invalid+=1;trace.append({'phase':'repair','step':step,'raw':raw,'unexecuted_error':str(exc)})
                    if invalid>cfg['max_repairs']:error='Exceeded format/schema repair budget';break
                    messages.extend([{'role':'assistant','content':raw},{'role':'user','content':'FORMAT_OR_SCHEMA_ERROR='+str(exc)+'. Chưa thực thi. Sửa đúng một JSON.'}]);continue
                kind=tools[action['name']].get('kind','unknown')
                if guard:
                    cached,previous=guard.lookup(action,kind)
                    if cached:
                        trace.append({'phase':'cached_read','step':step,'call':action,'executed':False})
                        messages.extend([{'role':'assistant','content':raw},{'role':'user','content':'DUPLICATE_READ_NOT_EXECUTED. Dùng kết quả đã có; chưa có lệnh không-phải-đọc xen giữa. PREVIOUS_RESULT='+dumps(previous)}])
                        if guard.blocks>=3:error='Repeated identical read loop stopped';break
                        continue
                result=env.call(action['name'],action['arguments'])
                if guard:guard.record(action,result,kind)
                trace.append({'phase':'tool','step':step,'call':action,'result':result})
                add_observation(messages,action,result)
                if app in window.active:window.active.remove(app);window.active.append(app)
                if action['name']=='sys.ask_missing_info' and isinstance(result,dict) and result.get('ok') is not False:asked=True
                if cfg.get('menu_recovery') and kind=='read' and empty_search(result) and empty_hints.get(action['name'],0)<2:
                    empty_hints[action['name']]=empty_hints.get(action['name'],0)+1
                    messages.append({'role':'user','content':'Kết quả đọc rỗng. Nếu phù hợp với yêu cầu và schema, thử từ khóa chính ngắn hơn hoặc công cụ liệt kê. '
                        'Không lặp đúng lời gọi cũ, không đoán ID; không tự đổi ngày/số tiền. Nếu vẫn thiếu thông tin bắt buộc thì hỏi lại. Có thể mở app khác trong menu nếu cần.'})
                    trace.append({'phase':'empty_result_guidance','step':step,'executed':False})
            else:error='Reached max_agent_steps without done'
    except (RuntimeError,ValueError,TypeError,KeyError) as exc:
        if 'out of memory' in str(exc).lower():raise
        error=type(exc).__name__+': '+str(exc)
    return {'id':str(item['id']),'tool_calls':env.get_tool_calls(),'state_hash':env.get_state_hash(),
        'trace':trace,'error':error,'invalid_model_actions':invalid,'context_trims':trims,'thinking':thinking,
        'router':{'initial_apps':initial,'final_apps':sorted(window.ever),'active_apps':window.active,
                  'menu_apps':sorted(window.apps),'menu_complete':True,'schema_evictions':window.evictions,
                  'fallback_all_apps':False,'fallback_menu_only':fallback,'format_error':route_error,'qwen_calls':calls,
                  'completion_reviews':reviews}}
