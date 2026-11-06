"""Online preparation only; pin resolved Hub revisions in each downloaded folder."""
import argparse
import json
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download

PROFILES = {
    'baseline': ['Qwen/Qwen3-4B-Instruct-2507', 'vinai/PhoWhisper-small'],
    'mentor': ['Qwen/Qwen3-4B-Instruct-2507', 'vinai/PhoWhisper-large'],
    'smoke': ['Qwen/Qwen3-0.6B', 'vinai/PhoWhisper-small'],
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile', choices=PROFILES, default='baseline')
    p.add_argument('--repo', action='append', help='Explicit repository; repeat to download multiple models')
    p.add_argument('--revision', default='main', help='Prefer a commit hash for reproducible teammate handoff')
    p.add_argument('--output', type=Path, default=Path('models'))
    p.add_argument('--tokenizer-only', action='store_true', help='Download config/tokenizer metadata, without weights')
    args = p.parse_args()
    for repo in args.repo or PROFILES[args.profile]:
        info = HfApi().model_info(repo, revision=args.revision)
        files = [f.rfilename for f in info.siblings]
        patterns = ['*.json', '*.txt', '*.model', '*.tiktoken', '*.jinja', 'README.md']
        if not args.tokenizer_only:
            patterns += ['*.safetensors'] if any(f.endswith('.safetensors') for f in files) else ['pytorch_model*.bin']
        destination = args.output / repo.split('/')[-1]
        snapshot_download(repo_id=repo, revision=info.sha, local_dir=destination, allow_patterns=patterns)
        (destination / 'download_manifest.json').write_text(json.dumps({
            'repo_id': repo, 'revision': info.sha, 'tokenizer_only': args.tokenizer_only}, indent=2), encoding='utf-8')
        print(f'{repo}@{info.sha} -> {destination}')


if __name__ == '__main__':
    main()
