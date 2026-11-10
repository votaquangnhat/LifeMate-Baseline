"""Separate Colab inference runtime; preserve host CUDA, pin teammate Python packages."""
import json
import re
import subprocess
import sys
from pathlib import Path
from textwrap import dedent


def pinned_packages(hints, defaults):
    # Only known packages and official Unsloth URLs are admitted from export metadata.
    result=[]
    for requirement in defaults:
        name=re.split(r'[ @<>=!~]',requirement)[0]
        if name in ('unsloth','unsloth_zoo'):
            commit=hints.get('source_revisions',{}).get(name,{}).get('vcs_info',{}).get('commit_id')
            if not isinstance(commit,str) or not re.fullmatch('[0-9a-f]{40}',commit):
                raise ValueError('Missing saved Unsloth source revision for '+name)
            repo='unsloth' if name=='unsloth' else 'unsloth-zoo'
            result.append(f'{name} @ git+https://github.com/unslothai/{repo}.git@{commit}')
        elif name in hints:
            version=hints[name]
            if not isinstance(version,str) or not re.fullmatch(r'[0-9][A-Za-z0-9.+_-]*',version):
                raise ValueError('Invalid saved package version: '+name)
            result.append(name+'=='+version)
        else:result.append(requirement)
    return result


def setup_environment(WORK_ROOT, OUTPUT_DIR, package_hints=None):
    import os
    import sys
    import json
    import subprocess
    from pathlib import Path


    def prepare_agent_python(env_dir, host_python):
        env_dir = Path(env_dir)
        env_dir.mkdir(parents=True, exist_ok=True)
        # Always repair/revalidate: a failed ensurepip run can leave bin/python behind.
        # --without-pip avoids invoking ensurepip at all; no apt/get-pip.py required.
        subprocess.run([str(host_python), '-m', 'venv', '--without-pip',
                        '--system-site-packages', str(env_dir)], check=True)
        agent_python = env_dir / 'bin/python'
        prefix = json.loads(subprocess.check_output(
            [str(agent_python), '-c', 'import sys,json; print(json.dumps({"prefix":sys.prefix,"base":sys.base_prefix}))'],
            text=True))
        if Path(prefix['prefix']).resolve() != env_dir.resolve() or prefix['prefix'] == prefix['base']:
            raise RuntimeError('The agent interpreter is not inside the intended environment.')
        pip_env = os.environ.copy()
        pip_command = [str(host_python), '-m', 'pip']
        help_text = subprocess.check_output(pip_command + ['--help'], text=True, env=pip_env)
        if '--python' not in help_text:
            # Compatibility for a very old host pip; keep the upgrade local.
            bootstrap = env_dir / 'pip-bootstrap'
            subprocess.run(pip_command + ['install', '--upgrade', '--target', str(bootstrap), 'pip>=22.3'],
                           check=True, env=pip_env)
            pip_env['PYTHONPATH'] = str(bootstrap) + os.pathsep + pip_env.get('PYTHONPATH', '')
        # Host pip can manage a target environment even when it has no pip installed.
        pip_command += ['--python', str(agent_python)]
        subprocess.run(pip_command + ['--version'], check=True, env=pip_env)
        print('Verified agent interpreter:', agent_python, flush=True)
        return agent_python, pip_command, pip_env

    # WORK_ROOT defined above
    ENV_DIR = WORK_ROOT / 'agent_env'
    ENV_DIR.mkdir(parents=True, exist_ok=True)
    AGENT_PYTHON = ENV_DIR / 'bin/python'
    probe_code = dedent('''
    import torch, json, importlib.metadata as m
    assert torch.cuda.is_available() and torch.cuda.device_count() >= 1, 'Select a Colab GPU runtime.' 
    pinned={}
    for name in ('torch','torchvision','torchaudio','triton'):
     try: pinned[name]=m.version(name)
     except m.PackageNotFoundError: pass
    print(json.dumps({'pinned': pinned, 'gpus': [{'name':torch.cuda.get_device_name(i),'gib':torch.cuda.get_device_properties(i).total_memory/2**30} for i in range(torch.cuda.device_count())]}))
    ''')
    host = json.loads(subprocess.check_output([sys.executable, '-c', probe_code], text=True))
    print('Existing CUDA stack:', json.dumps(host, indent=2))
    AGENT_PYTHON, AGENT_PIP, AGENT_PIP_ENV = prepare_agent_python(ENV_DIR, sys.executable)
    constraints = ENV_DIR / 'cuda-constraints.txt'
    constraints.write_text(''.join(f'{name}=={version}\n' for name, version in host['pinned'].items()))
    PACKAGES = [line.strip() for line in (Path(__file__).parent / 'requirements.txt').read_text().splitlines()
                if line.strip() and not line.lstrip().startswith('#')]
    PACKAGES += ['llmlingua>=0.2.2','nltk','tiktoken','tokenizers','huggingface_hub','safetensors']
    PACKAGES = ["transformers>=5.2,<6" if x.startswith("transformers") else x for x in PACKAGES]
    if package_hints:
        PACKAGES = pinned_packages(package_hints, PACKAGES)
    # Reproduce package versions/Unsloth commits when resuming this run.
    saved_path = OUTPUT_DIR / 'package_versions.json'
    saved_versions = json.loads(saved_path.read_text()) if saved_path.is_file() else None
    if saved_versions:
        if saved_versions['torch'] != host['pinned']['torch']:
            raise RuntimeError('Colab CUDA stack changed; use a new run or restore the original runtime.')
        PACKAGES = [f'{name}=={saved_versions[name]}' for name in
                    ('transformers','peft','accelerate','bitsandbytes','datasets','trl','tokenizers',
                     'jinja2','huggingface_hub','safetensors','soundfile','librosa','sentencepiece','llmlingua','nltk','tiktoken')]
        for name, url in [('unsloth','https://github.com/unslothai/unsloth.git'),
                          ('unsloth_zoo','https://github.com/unslothai/unsloth-zoo.git')]:
            commit = saved_versions.get('source_revisions',{}).get(name,{}).get('vcs_info',{}).get('commit_id')
            PACKAGES.append(f'{name} @ git+{url}@{commit}' if commit else f'{name}=={saved_versions[name]}')
    # Resolve compatible library versions; preserve the existing torch/CUDA builds.
    subprocess.run(AGENT_PIP + ['install', '--upgrade',
                    '--upgrade-strategy', 'only-if-needed', '-c', str(constraints), *PACKAGES], check=True, env=AGENT_PIP_ENV)
    version_code = dedent('''
    import importlib.metadata as m, json, torch
    names=['torch','transformers','peft','accelerate','bitsandbytes','datasets','trl','unsloth','unsloth_zoo','tokenizers','jinja2','huggingface_hub','safetensors','soundfile','librosa','sentencepiece','llmlingua','nltk','tiktoken']
    result={name:m.version(name) for name in names}
    result['source_revisions']={}
    for name in ['unsloth','unsloth_zoo']:
     raw=m.distribution(name).read_text('direct_url.json')
     if raw:
      info=json.loads(raw)
      result['source_revisions'][name]={'url':info.get('url'),'vcs_info':{'commit_id':info.get('vcs_info',{}).get('commit_id')}}
    print(json.dumps(result))
    ''')
    versions = json.loads(subprocess.check_output([str(AGENT_PYTHON), '-c', version_code], text=True))
    if versions['torch'] != host['pinned']['torch']:
        raise RuntimeError('Unexpected CUDA stack change. Inspect install output before loading a model.')
    print('Agent environment:', json.dumps(versions, indent=2))
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    if saved_versions and versions != saved_versions:
        raise RuntimeError('Package versions differ from the saved run. Inspect before resuming.')
    (OUTPUT_DIR / 'package_versions.json').write_text(json.dumps(versions, indent=2))

    return AGENT_PYTHON
