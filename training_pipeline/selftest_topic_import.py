"""Import and host workflow contracts; no GPU or model downloads."""
import json
import ast
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch
from topic_artifacts import import_export,validated_config
from topic_environment import pinned_packages
from topic_workflow import configure,choose_mode,compare_validation,previous_winner,previous_transcripts


def fixture_config():
    cfg=dict(model_family='qwen3_5',history_router=True,thinking_mode=True,
             lora_rank=8,lora_alpha=16,lora_dropout=0.0,session_deadline_unix=1,
             model_id='unsloth/Qwen3.5-4B',tokenizer_id='Qwen/Qwen3.5-4B',
             max_length=4096,max_length_cap=4096,requested_max_length=4096,
             history_token_budget=1024,router_prompt_tokens=2048,executor_prompt_tokens=2560)
    for prefix in ('model','tokenizer','reranker','history_compressor','segmenter_model','segmenter_tokenizer'):
        cfg.setdefault(prefix+'_id',prefix);cfg[prefix+'_revision']='a'*40
    return cfg


def create_archive(root,alter=None):
    cfg=fixture_config()
    if alter:cfg.update(alter)
    content={'run_config.json':json.dumps(cfg),'model_revisions.json':json.dumps({cfg[k]:cfg[k[:-3]+'_revision'] for k in cfg if k.endswith('_id')}),
             'package_versions.json':'{}','prepared_fingerprint.json':'{}',
             'best_adapter/adapter_config.json':json.dumps(dict(r=8,lora_alpha=16,lora_dropout=0.0,peft_type='LORA')),
             'best_adapter/adapter_model.safetensors':'fixture weights',
             'best_adapter/tokenizer.json':'{}','best_adapter/tokenizer_config.json':'{"chat_template":"template"}',
             'code/never_execute.py':'raise RuntimeError("bundled code must not run")'}
    archive=root/'best_adapter.zip'
    with zipfile.ZipFile(archive,'w') as z:
        for k,v in content.items():z.writestr(k,v)
    return archive


class Tests(unittest.TestCase):
    def test_import_uses_data_not_bundled_code_and_detects_mutation(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);a=create_archive(root);out=import_export(a,root/'import')
            self.assertFalse((out/'code').exists())
            self.assertEqual(import_export(a,out),out)
            (out/'best_adapter/adapter_model.safetensors').write_text('different')
            with self.assertRaisesRegex(ValueError,'changed or incomplete'):import_export(a,out)

    def test_path_escape_and_old_adapter_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);a=create_archive(root)
            with zipfile.ZipFile(a,'a') as z:z.writestr('../escape','bad')
            with self.assertRaisesRegex(ValueError,'Unsafe'):import_export(a,root/'import')
            self.assertFalse((root/'escape').exists())
            a=create_archive(root,{'history_router':False})
            with self.assertRaisesRegex(ValueError,'topical-router'):import_export(a,root/'other')

    def test_config_preserves_adapter_protocol_and_removes_kaggle_deadlines(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);imported=import_export(create_archive(root),root/'import')
            exp=root/'experiment';exp.mkdir();(exp/'package_versions.json').write_text('{}')
            session=dict(imported=str(imported),experiment=str(exp),data=str(root/'data'),code='code',work=str(root/'work'))
            baseline=configure(session,'baseline');enhanced=configure(session,'enhanced')
            self.assertEqual(baseline['lora_rank'],8)
            self.assertEqual(baseline['executor_prompt_tokens'],2560)
            self.assertEqual(enhanced['max_length'],8192)
            self.assertEqual(enhanced['lora_rank'],8)
            self.assertEqual(enhanced['rerank_gpu'],0)
            self.assertNotIn('session_deadline_unix',enhanced)
            self.assertEqual(configure(session,'enhanced'),enhanced)
            self.assertEqual(json.loads((imported/'run_config.json').read_text())['session_deadline_unix'],1)
            with self.assertRaises(FileNotFoundError):choose_mode(session,'auto')

    def test_comparison_selects_validation_winner_and_baseline_on_ties(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);imported=import_export(create_archive(root),root/'import')
            for name in ('validation_references.jsonl','split_assignments.json'):(imported/name).write_text('{}')
            exp=root/'experiment';exp.mkdir();(exp/'package_versions.json').write_text('{}');(exp/'experiment_source.json').write_text('{}')
            session=dict(imported=str(imported),experiment=str(exp),data=str(root/'data'),code='code',work=str(root/'work'))
            scores={'baseline':60,'enhanced':61,'read_guard':59}
            def evaluate(session,cfg,part):
                self.assertEqual(part,'validation')
                (Path(cfg['output_dir'])/'validation_metrics.json').write_text(json.dumps({
                    'adapter_sha256':'same','samples':77,'official_score_100':scores[cfg['agent_variant']]}))
            with patch('topic_workflow.run_part',side_effect=evaluate):
                report=compare_validation(session)
                self.assertEqual(report['chosen'],'enhanced');self.assertEqual(choose_mode(session,'auto'),'enhanced')
                scores['enhanced']=60
                self.assertEqual(compare_validation(session)['chosen'],'baseline')

    def test_notebook_configuration_and_independent_cell_are_complete(self):
        notebook=Path(__file__).resolve().parent.parent/'notebooks/teammate_topic_agent_inference.ipynb'
        cells={c.get('id'):''.join(c['source']) for c in json.loads(notebook.read_text())['cells']}
        for name in ('mount','public'):
            tree=ast.parse(cells[name]);scope={'Path':Path}
            required={'DRIVE_ROOT','IMPORT_NAME','EXPERIMENT_NAME','ADAPTER_ZIP','REPORTS_ZIP','WORK_ROOT'}
            if name=='public':required|={'MODE','ASR_MODE','ASR_CHECKPOINT','ASR_DECODER','TRANSCRIPTS_PATH'}
            nodes=[n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in required for t in n.targets)]
            exec(compile(ast.Module(body=nodes,type_ignores=[]),'<notebook configuration>','exec'),scope)
            self.assertTrue(required<=scope.keys())
            self.assertEqual(scope['ADAPTER_ZIP'].name,'best_adapter.zip')
        for c in json.loads(notebook.read_text())['cells']:
            if c['cell_type']=='code':ast.parse(''.join(c['source']))

    def test_menu_variants_keep_weights_and_validate_reference_and_asr(self):
        from topic_artifacts import digest
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);imported=import_export(create_archive(root),root/'import')
            exp=root/'runs'/'new';exp.mkdir(parents=True);(exp/'package_versions.json').write_text('{}')
            session=dict(root=str(root),imported=str(imported),experiment=str(exp),data=str(root/'data'),code='code',work=str(root/'work'))
            cfg=configure(session,'menu_recovery')
            self.assertEqual(cfg['model_id'],'unsloth/Qwen3.5-4B')
            self.assertEqual(cfg['lora_rank'],8);self.assertTrue(cfg['preserve_full_schemas'])
            self.assertTrue(cfg['menu_recovery']);self.assertTrue(cfg['thinking_mode'])
            prior=root/'runs'/'old';prior.mkdir()
            write={'chosen':'enhanced','metrics':{'enhanced':{'adapter_sha256':digest(imported/'best_adapter/adapter_model.safetensors')}}}
            (prior/'agent_comparison.json').write_text(json.dumps(write))
            self.assertEqual(previous_winner(session,'old'),'enhanced')
            directory=prior/'enhanced';directory.mkdir();asr=root/'asr.jsonl';asr.write_text('{}')
            (directory/'run_config.json').write_text(json.dumps({'public_transcripts_path':str(asr)}))
            (directory/'public_rerank_fingerprint.json').write_text(json.dumps({'transcripts':digest(asr)}))
            self.assertEqual(previous_transcripts(session,'old','enhanced'),str(asr))
            asr.write_text('different')
            with self.assertRaisesRegex(ValueError,'cannot be verified'):previous_transcripts(session,'old','enhanced')
            write['metrics']['enhanced']['adapter_sha256']='different'
            (prior/'agent_comparison.json').write_text(json.dumps(write))
            with self.assertRaisesRegex(ValueError,'different weights'):previous_winner(session,'old')

    def test_new_menu_notebook_defines_its_independent_inputs(self):
        notebook=Path(__file__).resolve().parent.parent/'notebooks/menu_agent_inference.ipynb'
        cells={c.get('id'):''.join(c['source']) for c in json.loads(notebook.read_text())['cells']}
        needed={'DRIVE_ROOT','IMPORT_NAME','PREVIOUS_EXPERIMENT','REFERENCE_MODE','EXPERIMENT_NAME',
                'ADAPTER_ZIP','REPORTS_ZIP','WORK_ROOT'}
        for name in ('mount','public'):
            names=needed | ({'MODE','ASR_MODE','TRANSCRIPTS_PATH','ASR_CHECKPOINT','ASR_DECODER'} if name=='public' else set())
            tree=ast.parse(cells[name]);scope={'Path':Path}
            nodes=[n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in names for t in n.targets)]
            exec(compile(ast.Module(body=nodes,type_ignores=[]),'<menu notebook>','exec'),scope)
            self.assertTrue(names<=scope.keys())
            self.assertEqual(scope['IMPORT_NAME'],'teammate_topic_20261109')
            if name=='public':self.assertEqual(scope['ASR_MODE'],'transcripts')
        for c in json.loads(notebook.read_text())['cells']:
            if c['cell_type']=='code':ast.parse(''.join(c['source']))

    def test_package_hints_cannot_supply_arbitrary_requirement_or_url(self):
        hints={'transformers':'5.17.0','source_revisions':{'unsloth':{'url':'https://untrusted.invalid','vcs_info':{'commit_id':'b'*40}}}}
        result=pinned_packages(hints,['transformers>=5.2,<6','unsloth @ placeholder'])
        self.assertEqual(result[0],'transformers==5.17.0')
        self.assertIn('https://github.com/unslothai/unsloth.git@',result[1])
        hints['transformers']='5.17.0 --extra-index-url bad'
        with self.assertRaises(ValueError):pinned_packages(hints,['transformers'])


if __name__=='__main__':unittest.main(verbosity=2)
