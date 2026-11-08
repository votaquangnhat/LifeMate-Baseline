"""CPU contracts for DDP launches, effective batch and stress-case sharding."""
import unittest
from distributed_utils import launch_spec, training_plan, smoke_dataset


CFG = {"num_gpus":2, "eval_num_gpus":2, "rerank_gpu":1, "code_dir":"/tmp/code",
       "config_path":"/tmp/config.json", "epochs":1, "per_gpu_batch_size":1,
       "gradient_accumulation_steps":4}


class Tests(unittest.TestCase):
    def test_train_uses_two_torchrun_processes(self):
        command, devices = launch_spec(CFG, "train_agent.py", [])
        self.assertIn("torch.distributed.run", command)
        self.assertEqual(command[command.index("--nproc_per_node")+1], "2")
        self.assertEqual(devices, "0,1")

    def test_preflight_uses_same_distributed_launch(self):
        train, a = launch_spec(CFG,"train_agent.py",[])
        smoke, b = launch_spec(CFG,"train_agent.py",["--smoke"])
        self.assertEqual(smoke, train+["--smoke"])
        self.assertEqual(a,b)

    def test_reranker_is_single_process_on_its_selected_gpu(self):
        for script in ("prepare_agent.py", "prepare_public.py"):
            command, devices = launch_spec(CFG,script,[])
            self.assertNotIn("torch.distributed.run",command)
            self.assertEqual(devices,"1")

    def test_rollout_gpu_count_is_independent_of_training(self):
        cfg={**CFG,"eval_num_gpus":1}
        command, devices=launch_spec(cfg,"rollout_agent.py",[])
        self.assertNotIn("torch.distributed.run",command)
        self.assertEqual(devices,"0")
        self.assertIn("torch.distributed.run",launch_spec(cfg,"train_agent.py",[])[0])

    def test_global_batch_matches_previous_single_gpu_recipe(self):
        before={**CFG,"num_gpus":1,"gradient_accumulation_steps":8}
        a,b=training_plan(before,4062),training_plan(CFG,4062)
        self.assertEqual(a["effective_batch"],b["effective_batch"])
        self.assertEqual(b["effective_batch"],8)
        self.assertEqual(a["updates_per_epoch"],b["updates_per_epoch"])
        self.assertEqual(b["train_gpus"],2)

    def test_batch_accounting_handles_partial_final_batches(self):
        self.assertEqual(training_plan(CFG,9)["updates_per_epoch"],2)
        self.assertEqual(training_plan(CFG,8)["updates_per_epoch"],1)
        self.assertEqual(training_plan({**CFG,"gradient_accumulation_steps":8},16)["effective_batch"],16)

    def test_every_rank_sees_every_stress_case_for_full_accumulation(self):
        for batch in (1,2):
            cfg={**CFG,"per_gpu_batch_size":batch}
            cases=[{"case":i} for i in range(5)]
            rows=smoke_dataset(cases,cfg)
            batches=[rows[i:i+batch] for i in range(0,len(rows),batch)]
            for rank in range(2):
                local=batches[rank::2]
                updates=[local[i:i+4] for i in range(0,len(local),4)]
                self.assertEqual(len(updates),5)
                for case,update in enumerate(updates):
                    self.assertEqual(len(update),4)
                    self.assertTrue(all(row["case"]==case for micro in update for row in micro))

    def test_invalid_gpu_count_is_rejected(self):
        with self.assertRaises(ValueError):
            launch_spec({**CFG,"num_gpus":0},"train_agent.py",[])
        with self.assertRaises(ValueError):
            training_plan({**CFG,"gradient_accumulation_steps":0},100)


if __name__ == "__main__":
    unittest.main(verbosity=2)
