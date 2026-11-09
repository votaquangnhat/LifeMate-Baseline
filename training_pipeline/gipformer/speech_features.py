"""Gipformer release-compatible float-waveform fbank; NOT ChunkFormer's PCM-scale CMVN."""
import math
import soundfile as sf
import torch
import torchaudio
import kaldifeat

FEATURE_VERSION = 'gip_fbank80_float_snipfalse_hfminus400_v1'
_FBANK = None

def ctc_output_frames(frames):
    # Historical function name retained for manifest code. Gipformer has no CTC
    # branch here: (T-7)//2 frontend, then ceil-divide by 2 in Zipformer.
    return max(0, (((int(frames)-7)//2)+1)//2)

def fbank():
    global _FBANK
    if _FBANK is None:
        opts = kaldifeat.FbankOptions()
        opts.device = torch.device('cpu')
        opts.frame_opts.dither = 0
        opts.frame_opts.snip_edges = False
        opts.frame_opts.samp_freq = 16000
        opts.mel_opts.num_bins = 80
        opts.mel_opts.high_freq = -400
        _FBANK = kaldifeat.Fbank(opts)
    return _FBANK

def extract_feature(path, speed=1.0, noise_snr=None, noise_seed=0):
    wav, sr = sf.read(str(path), dtype='float32', always_2d=True)
    wav = torch.from_numpy(wav).mean(1).contiguous()
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    seconds = len(wav)/16000
    if speed != 1.0:
        wav = torchaudio.functional.resample(wav, round(16000*speed), 16000)
    if noise_snr is not None:
        g = torch.Generator().manual_seed(int(noise_seed))
        noise = torch.randn(wav.shape, generator=g)
        signal_rms = wav.square().mean().sqrt().clamp_min(1e-6)
        noise = noise/noise.square().mean().sqrt().clamp_min(1e-6)
        wav = wav + noise*signal_rms/(10**(float(noise_snr)/20))
        # Rescale the complete mixture only if needed; this preserves its SNR.
        wav = wav/wav.abs().max().clamp_min(1.0)
    if not torch.isfinite(wav).all() or len(wav) < 400:
        raise ValueError('Invalid audio: ' + str(path))
    with torch.no_grad():
        features = fbank()([wav])[0]
    if not torch.isfinite(features).all():
        raise ValueError('Nonfinite fbank: ' + str(path))
    return features, seconds

