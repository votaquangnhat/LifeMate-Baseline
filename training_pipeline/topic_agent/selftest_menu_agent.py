"""Menu coverage, complete schema lifecycle and bounded recovery contracts."""
import copy
import json
import unittest
from agent_core import installed_tool_names,dumps,prompt_ids
from menu_agent import menu,router_prompt,SchemaWindow,execute,calendar_context,empty_search
from tool_router import clean_context
from selftest_agent import TOOLS,FakeEnv,item,call
from selftest_context import FakeTokenizer
from selftest_history_router import config


class Tests(unittest.TestCase):
    def setup(self,recovery=False):
        cfg=config('/tmp');cfg.update(menu_agent=True,menu_recovery=recovery,protected_context=True,preserve_full_schemas=True,
            max_length=20000,max_new_tokens=768,max_new_tokens_cap=768,router_prompt_tokens=16000,executor_prompt_tokens=18000,max_agent_steps=20)
        tools=copy.deepcopy(TOOLS);tools['contacts.search'].update(kind='read',returns='contacts with exact IDs',description='Complete lookup description')
        tools['contacts.search']['_app']={'name_vi':'Danh bạ','description':'Tra cứu người nhận'}
        obj=item();history,request,context=clean_context(obj,tools,cfg)
        ctx=dict(history=history,request=request,public_context=context,names=installed_tool_names(obj,tools,cfg['global_meta_tools']))
        obj['_router_context']=ctx
        return cfg,tools,obj,FakeTokenizer()

    def run_agent(self,answers,recovery=False):
        cfg,tools,obj,tok=self.setup(recovery)
        queue=iter(answers);seen=[]
        def generate(messages):seen.append(copy.deepcopy(messages));return next(queue)
        generate.tokenizer=tok
        env=FakeEnv();r=execute(obj,tools,env,generate,cfg)
        return r,env,seen

    def test_all_app_menu_has_vietnamese_names_but_no_parameter_schemas(self):
        cfg,tools,obj,tok=self.setup();ctx=obj['_router_context']
        rows=menu(ctx,tools)
        self.assertEqual({r['app'] for r in rows},{'contacts','sms'})
        self.assertEqual(rows[0]['name_vi'],'Danh bạ')
        text=dumps(router_prompt(ctx,tools,cfg,tok))
        self.assertNotIn('parameters',text);self.assertNotIn('SECRET_GOLD_STATE',text)

    def test_wrong_initial_router_choice_can_auto_open_and_regenerate(self):
        read={'name':'contacts.search','arguments':{'query':'An'}}
        r,env,seen=self.run_agent(['{"apps":["sms"]}',dumps(read),dumps(read),dumps(call()),'{"done":true}'])
        self.assertEqual(env.calls,[read,call()]);self.assertIsNone(r['error'])
        self.assertTrue(any(t['phase']=='auto_open' for t in r['trace']))
        self.assertEqual(r['router']['final_apps'],['contacts','sms'])
        self.assertIn('contacts with exact IDs',seen[2][0]['content'])

    def test_router_format_failure_keeps_menu_without_loading_all_schemas(self):
        r,env,seen=self.run_agent(['bad','bad','{"done":true}'])
        self.assertTrue(r['router']['fallback_menu_only'])
        system=seen[-1][0]['content'];schemas=json.loads(system.split('\nTOOLS=')[1].split('\nCONTEXT=')[0])
        self.assertEqual([s['name'] for s in schemas],['sys.ask_missing_info'])
        self.assertEqual(env.calls,[])
        self.assertIn('contacts',system)

    def test_schema_eviction_is_reversible_and_keeps_complete_parameters_returns(self):
        cfg,tools,obj,tok=self.setup();ctx=obj['_router_context']
        for name in ('contacts.search','sms.send'):tools[name]['description']='detail '*300
        window=SchemaWindow(ctx,tools,cfg,tok)
        messages=[{'role':'system','content':''},{'role':'user','content':'HISTORY=[]\nFINAL_REQUEST=test'}]
        single=max(len(prompt_ids(tok,window.render(messages,[a]))) for a in ('sms','contacts'))
        both=len(prompt_ids(tok,window.render(messages,['sms','contacts'])))
        cfg['executor_prompt_tokens']=(single+both)//2
        messages,_=window.fit(messages,['sms'])
        messages,a=window.fit(messages,['contacts'])
        self.assertEqual(window.active,['contacts']);self.assertEqual(a['evicted_apps'],['sms'])
        schemas=json.loads(messages[0]['content'].split('\nTOOLS=')[1].split('\nCONTEXT=')[0])
        lookup=next(s for s in schemas if s['name']=='contacts.search')
        self.assertEqual(lookup['description'],tools['contacts.search']['description'])
        self.assertEqual(lookup['parameters'],tools['contacts.search']['parameters'])
        self.assertEqual(lookup['returns'],'contacts with exact IDs')
        messages,a=window.fit(messages,['sms']);self.assertEqual(window.active,['sms'])
        self.assertIn('contacts',messages[0]['content'].split('\nTOOLS=')[0])

    def test_completion_review_can_recover_an_unopened_app_but_is_bounded(self):
        read={'name':'contacts.search','arguments':{'query':'An'}}
        r,env,seen=self.run_agent(['{"apps":["sms"]}',dumps(call()),'{"done":true}',
            '{"open_apps":["contacts"]}',dumps(read),'{"done":true}'],True)
        self.assertEqual(env.calls,[call(),read]);self.assertEqual(r['router']['completion_reviews'],1)
        self.assertIsNone(r['error'])
        r,env,_=self.run_agent(['{"apps":[]}','{"done":true}','{"done":true}'],True)
        self.assertEqual(env.calls,[]);self.assertEqual(r['router']['completion_reviews'],1)

    def test_calendar_uses_sample_timezone_not_machine_clock(self):
        ctx={'current_time':'2026-11-08T18:30:00+00:00','timezone':'Asia/Ho_Chi_Minh'}
        result=calendar_context(ctx)['date_reference']
        self.assertEqual(result['today'],'2026-11-09')
        self.assertEqual(result['next_week'],['2026-11-16','2026-11-22'])
        self.assertNotIn('date_reference',ctx)
        self.assertFalse(empty_search({'ok':False,'error':'permission denied'}))
        self.assertFalse(empty_search({'ok':True,'warnings':[]}))
        self.assertTrue(empty_search({'ok':True,'items':[]}))


if __name__=='__main__':unittest.main(verbosity=2)
