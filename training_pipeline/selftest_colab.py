import copy
import unittest
from asr_contract import merge_transcripts
from distributed_utils import launch_spec, training_plan

class Tests(unittest.TestCase):
    def setUp(self):
        self.items = [{'id':'public_1', 'audio':'audio/a.wav', 'context':{'x':1},
                       'audio_text':'GOLD MUST NOT LEAK', 'label':{'secret':True},
                       'history':[{'role':'user','audio':'audio/h.wav','audio_text':'OLD GOLD'},
                                  {'role':'assistant','tool_results':[{'id':'contact_1'}]}]}]
        self.rows = [{'id':'public_1','audio_text':'ASR request', 'history_audio_texts':['ASR history']}]
    def test_asr_replaces_all_user_text_and_preserves_context(self):
        before = copy.deepcopy(self.items)
        self.rows[0]['context'] = {'x':999}
        result = merge_transcripts(self.items,self.rows)[0]
        self.assertEqual(result['context'], {'x':1})
        self.assertEqual(result['history'][0]['audio_text'],'ASR history')
        self.assertEqual(result['history'][1], before[0]['history'][1])
        self.assertNotIn('label',result)
        self.assertEqual(self.items,before)
    def test_missing_history_never_uses_original(self):
        self.rows[0]['history_audio_texts'] = []
        with self.assertRaises(ValueError): merge_transcripts(self.items,self.rows)
    def test_duplicate_ids_rejected(self):
        with self.assertRaises(ValueError): merge_transcripts(self.items,self.rows * 2)
    def test_blank_final_rejected(self):
        self.rows[0]['audio_text'] = ' '
        with self.assertRaises(ValueError): merge_transcripts(self.items,self.rows)
    def test_audio_path_mapping(self):
        self.rows[0]['history_audio_texts'] = {'audio/h.wav':'mapped'}
        self.assertEqual(merge_transcripts(self.items,self.rows)[0]['history'][0]['audio_text'],'mapped')
    def test_single_gpu_launch_and_batch(self):
        cfg = dict(num_gpus=1,eval_num_gpus=1,rerank_gpu=0,code_dir='/code',config_path='/cfg',
                   per_gpu_batch_size=1,gradient_accumulation_steps=8,epochs=2)
        command, devices = launch_spec(cfg,'train_agent.py',[])
        self.assertNotIn('torch.distributed.run',command)
        self.assertEqual(devices,'0')
        self.assertEqual(training_plan(cfg,160)['effective_batch'],8)

if __name__ == '__main__': unittest.main(verbosity=2)
