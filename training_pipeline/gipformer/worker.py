"""Architecture/tokenizer only, extracted from teammate full-finetune worker."""
import argparse,json,sys,unicodedata
from pathlib import Path
import torch
from torch import nn

def norm(text):
    text = unicodedata.normalize('NFC', str(text)).lower()
    text = ''.join(c if unicodedata.category(c)[0] in 'LMN' else ' ' for c in text)
    return ' '.join(text.split())

class VocabTokenizer:
    """Use the released uppercase SentencePiece vocabulary and its native IDs."""
    VERSION = 'gipformer_sp_upper_v2'

    def __init__(self, path):
        import sentencepiece as spm
        self.path = Path(path)
        self.tokens = {}
        for line in self.path.read_text(encoding='utf-8').splitlines():
            piece, index = line.rsplit(maxsplit=1)
            if piece in self.tokens:
                raise ValueError('Duplicate released token: ' + piece)
            self.tokens[piece] = int(index)
        self.inverse = {v: k for k, v in self.tokens.items()}
        if len(self.inverse) != len(self.tokens):
            raise ValueError('Duplicate released token IDs')
        self.blank = self.tokens['<blk>']
        self.unk = self.tokens['<unk>']
        self.bpe_path = self.path.parent / 'bpe.model'
        self.sp = spm.SentencePieceProcessor(model_file=str(self.bpe_path))
        self.model_vocab_size = max(i for p, i in self.tokens.items() if not p.startswith('#')) + 1
        if self.blank != 0 or self.sp.unk_id() != self.unk:
            raise ValueError('Released Gipformer blank/unknown IDs do not match SentencePiece')
        if self.sp.get_piece_size() != self.model_vocab_size:
            raise ValueError('Released SentencePiece and model vocabulary sizes differ')
        mismatches = [(i, self.sp.id_to_piece(i), self.inverse.get(i))
                      for i in range(self.model_vocab_size)
                      if self.sp.id_to_piece(i) != self.inverse.get(i)]
        if mismatches:
            raise ValueError('SentencePiece IDs differ from tokens.txt: ' + repr(mismatches[:10]))
        self.special_ids = {i for p, i in self.tokens.items() if p.startswith(('<', '#'))}

    def input_text(self, text):
        # Gipformer's pretrained vocabulary is uppercase. Keep the original GT
        # and scoring normalization unchanged; only its tokenizer input is cased.
        return unicodedata.normalize('NFC', norm(text).upper())

    def encode(self, text):
        prepared = self.input_text(text)
        ids = list(self.sp.encode(prepared, out_type=int))
        pieces = self.sp.encode(prepared, out_type=str)
        if len(ids) != len(pieces):
            raise ValueError('SentencePiece ID/piece lengths differ')
        missing = [piece for piece, index in zip(pieces, ids) if index == self.unk]
        if any(i in self.special_ids for i in ids if i != self.unk):
            raise ValueError('Special/disambiguation symbol in training transcript')
        if not missing and norm(self.text(ids)) != norm(text):
            raise ValueError('Released tokenizer round-trip mismatch: ' + str(text))
        return ids, missing

    def text(self, ids):
        # Use SentencePiece's native decoder, as in the official inference code.
        # Repeated RNN-T tokens are meaningful: never apply CTC repeat collapse.
        values = [int(i) for i in ids]
        if any(i not in self.inverse for i in values):
            raise ValueError('Decoded token ID outside the released symbol table')
        values = [i for i in values if i not in self.special_ids or i == self.unk]
        return self.sp.decode(values).strip()

    def decode(self, ids):
        return self.text(ids)

    def smoke_test(self):
        examples = []
        for text in ('nếu mai mưa thì nhắn cho mẹ lúc sáu giờ',
                     'mạ ni hông mô răng', 'sjc iphone wifi',
                     'không chín tám ba hai một', 'bảy bảy bảy'):
            ids, missing = self.encode(text)
            if missing or not ids:
                raise ValueError('Tokenizer smoke test failed: ' + repr((text, missing)))
            examples.append(dict(text=text, tokenizer_input=self.input_text(text),
                                 ids=ids, decoded=self.text(ids)))
        return dict(version=self.VERSION, input_case='uppercase',
                    sentencepiece_ids_match_symbol_table=True,
                    vocab_size=self.model_vocab_size, examples=examples)

def runtime_vocab_path(cfg):
    return Path(cfg.get('vocab_path', str(Path(cfg['model_dir']) / 'tokens.txt')))

def setup_source(cfg):
    root = Path(cfg['source_dir'])
    for p in [root, root/'egs/librispeech/ASR', root/'egs/librispeech/ASR/zipformer']:
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))

class SpeechRNNT(nn.Module):
    """Full Gipformer: acoustic frontend/encoder, stateless decoder, joiner, and
    simple AM/LM projection heads used by native k2 pruned-RNN-T training.
    There is no newly initialized CTC or attention decoder in this recipe.
    """
    def __init__(self, cfg):
        super().__init__()
        setup_source(cfg)
        import k2
        from train import add_model_arguments, get_model, get_params, set_batch_count
        self.cfg, self.k2, self._set_batch_count = cfg, k2, set_batch_count
        parser = argparse.ArgumentParser(add_help=False)
        add_model_arguments(parser)
        params = get_params()
        params.update(vars(parser.parse_args([])))
        released = json.loads((Path(cfg['model_dir'])/'config.json').read_text())
        # The release's inference script uses icefall defaults. Accept explicit
        # architecture values from config when provided, never infer from labels.
        for key in list(params):
            if key in released and key not in ('vocab_size', 'blank_id', 'unk_id'):
                value = released[key]
                params[key] = ','.join(map(str, value)) if isinstance(params[key], str) and isinstance(value, list) else value
        tok = VocabTokenizer(runtime_vocab_path(cfg))
        params.update(feature_dim=80, blank_id=tok.blank, unk_id=tok.unk,
                      vocab_size=tok.model_vocab_size, context_size=int(released.get('context_size', 2)),
                      use_transducer=True, use_ctc=False, use_attention_decoder=False,
                      use_cr_ctc=False, causal=False, chunk_size='-1', left_context_frames='-1')
        self.asr = get_model(params)
        self.asr.unk_id = tok.unk
        self.params = dict(params)
        self.blank = tok.blank
        self.last_losses = {}
        self.set_schedule(0)

    def set_schedule(self, step):
        # Do not restart Zipformer's layer warmup at zero during fine-tuning.
        self._set_batch_count(self.asr, float(self.cfg.get('zipformer_schedule_start', 20000) + step))

    def encode(self, features, lengths):
        encoded, out_lengths = self.asr.forward_encoder(features, lengths)
        mask = torch.arange(encoded.shape[1], device=encoded.device)[None, :] < out_lengths[:, None]
        return encoded, mask[:, None, :], out_lengths

