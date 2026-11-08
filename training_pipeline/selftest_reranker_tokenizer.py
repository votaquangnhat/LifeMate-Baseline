"""Real Transformers tokenizer regression tests; no model weights or GPU needed."""
import contextlib
import json
import math
import sqlite3
import tempfile
import unittest
from pathlib import Path

import transformers
from transformers import AutoTokenizer, PreTrainedTokenizerFast, XLMRobertaTokenizer
from context_rerank import (CachedReranker, encode_rerank_pairs, rerank_pair_batches,
                            check_reranker_tokenizer)


def tokenizer_fixture():
    # Use the actual XLMRobertaTokenizer class on Transformers 5, the class from
    # the failure log. A tiny local vocabulary keeps this test offline.
    special = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
    alphabet = sorted(set("▁abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 _-.,!?" +
                          "Nhắc tôi gọi cho mẹ ngày mai Tạo nhắc việc với giờ và tiêu đề Tìm số điện thoại của An liên hệ theo tên trả về ID"))
    vocab = [(s,0.0) for s in special] + [(s,-1.0) for s in alphabet]
    if int(transformers.__version__.split(".")[0]) >= 5:
        return XLMRobertaTokenizer(vocab=vocab, model_max_length=512)
    from tokenizers import Tokenizer, decoders, pre_tokenizers, processors
    from tokenizers.models import Unigram
    backend = Tokenizer(Unigram(vocab=vocab, unk_id=3))
    backend.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁", prepend_scheme="always")
    backend.decoder = decoders.Metaspace(replacement="▁", prepend_scheme="always")
    backend.post_processor = processors.TemplateProcessing(single="<s> $A </s>",
        pair="<s> $A </s> </s> $B </s>", special_tokens=[("<s>",0),("</s>",2)])
    return PreTrainedTokenizerFast(tokenizer_object=backend, bos_token="<s>", cls_token="<s>",
                                   eos_token="</s>", sep_token="</s>", unk_token="<unk>", pad_token="<pad>")


CFG = {"rerank_max_length":512, "rerank_query_tokens":160, "rerank_batch_size":3}


class TensorStub:
    def __init__(self, values): self.values=values
    def to(self, device): return self
    def reshape(self, *shape): return self
    def float(self): return self
    def cpu(self): return self
    def tolist(self): return self.values


class ActualTokenizerCPU:
    """Use actual tokenization; substitute only the tensor container, not its IDs."""
    def __init__(self, tokenizer): self.tokenizer=tokenizer
    def __getattr__(self, name):
        if name == "prepare_for_model":
            raise AssertionError("The legacy API must never be called")
        return getattr(self.tokenizer,name)
    def __call__(self, **kwargs):
        tensor_type=kwargs.pop("return_tensors",None)
        values=self.tokenizer(**kwargs,return_tensors=None)
        return {k:TensorStub(v) for k,v in values.items()} if tensor_type else values


class ModelStub:
    def __init__(self): self.calls=0
    def __call__(self, input_ids, **kwargs):
        self.calls+=1
        scores=[sum(row)/10000 for row in input_ids.values]
        return type("Outputs",(),{"logits":TensorStub(scores)})()


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.tokenizer=tokenizer_fixture()

    def test_short_vietnamese_pairs_equal_native_tokenizer(self):
        pairs=[("Nhắc tôi gọi cho mẹ ngày mai","Tạo nhắc việc với ngày giờ"),
               ("Tìm số điện thoại của An","Tìm liên hệ theo tên")]
        actual=encode_rerank_pairs(self.tokenizer,pairs,512)
        expected=self.tokenizer(text=[q for q,d in pairs],text_pair=[d for q,d in pairs],
            padding=True,truncation=True,max_length=512,return_attention_mask=True)
        self.assertEqual(actual["input_ids"],expected["input_ids"])
        self.assertEqual(actual["attention_mask"],expected["attention_mask"])

    def test_special_tokens_and_padding_masks(self):
        batch=encode_rerank_pairs(self.tokenizer,[("An","ID"),("Nhắc tôi gọi cho mẹ ngày mai","Tạo nhắc việc")],512)
        for ids,mask in zip(batch["input_ids"],batch["attention_mask"]):
            live=[token for token,m in zip(ids,mask) if m]
            self.assertEqual(live[0],self.tokenizer.bos_token_id)
            self.assertEqual(live[-1],self.tokenizer.eos_token_id)
            self.assertEqual(live.count(self.tokenizer.eos_token_id),3)
        self.assertIn(0,batch["attention_mask"][0])

    def test_long_query_documents_are_windowed_in_bounded_batches(self):
        query="nhắc gọi "*100
        docs=["liên hệ An "*100+" LAST_MARKER", "", "ngày mai "*100]
        owners=set();pair_count=0;tail=False
        for pairs,indices in rerank_pair_batches(self.tokenizer,query,docs,range(len(docs)),CFG):
            self.assertLessEqual(len(pairs),CFG["rerank_batch_size"])
            self.assertEqual(len(pairs),len(indices))
            batch=encode_rerank_pairs(self.tokenizer,pairs,512)
            self.assertTrue(all(len(ids)<=512 for ids in batch["input_ids"]))
            owners.update(indices);pair_count+=len(pairs)
            tail |= any("LAST_MARKER" in d for _,d in pairs)
        self.assertEqual(owners,{0,1,2})
        self.assertGreater(pair_count,len(docs))
        self.assertTrue(tail)

    def test_empty_texts_still_form_valid_pairs(self):
        batches=list(rerank_pair_batches(self.tokenizer,"",[""],[0],CFG))
        self.assertEqual(len(batches),1)
        encoded=encode_rerank_pairs(self.tokenizer,batches[0][0],512)
        self.assertEqual(len(encoded["input_ids"][0]),4)

    def test_probe_and_local_save_reload_use_real_tokenizer(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.tokenizer.save_pretrained(tmp)
            loaded=AutoTokenizer.from_pretrained(tmp,local_files_only=True)
            report=check_reranker_tokenizer(loaded,512)
            self.assertTrue(report["passed"])
            self.assertEqual(report["pair_api"],"__call__(text, text_pair)")

    def test_score_and_cache_with_actual_token_ids(self):
        scorer=CachedReranker.__new__(CachedReranker)
        scorer.cfg=CFG
        scorer.tokenizer=ActualTokenizerCPU(self.tokenizer)
        scorer.device="cpu"
        scorer.torch=type("TorchStub",(),{"inference_mode":staticmethod(contextlib.nullcontext)})()
        scorer.signature={"regression":1}
        scorer.db=sqlite3.connect(":memory:")
        scorer.db.execute("CREATE TABLE scores (key TEXT PRIMARY KEY, score REAL NOT NULL)")
        scorer.model=ModelStub()
        docs=["Tạo nhắc việc ngày mai","Tìm liên hệ An", "Tạo nhắc việc "*100]
        first=scorer.score("Nhắc tôi gọi cho mẹ ngày mai",docs)
        calls=scorer.model.calls
        self.assertEqual(len(first),len(docs))
        self.assertTrue(all(math.isfinite(v) for v in first))
        self.assertGreater(calls,0)
        second=scorer.score("Nhắc tôi gọi cho mẹ ngày mai",docs)
        self.assertEqual(first,second)
        self.assertEqual(scorer.model.calls,calls)
        scorer.close()

    def test_invalid_pair_batches_fail_clearly(self):
        with self.assertRaises(ValueError): encode_rerank_pairs(self.tokenizer,[],512)
        with self.assertRaises(ValueError):
            list(rerank_pair_batches(self.tokenizer,"An",["ID"],[0],{**CFG,"rerank_batch_size":0}))


if __name__ == "__main__":
    print("Real tokenizer regression tests; transformers="+transformers.__version__,flush=True)
    unittest.main(verbosity=2)
