import os,sys,json,shutil,subprocess
from pathlib import Path
from .processes import run_jobs
from .submission_utils import write_json

def run(command, name, cfg, env=None):
    run_jobs([(name,command,env or os.environ.copy(),None)],Path(cfg['output_dir'])/'logs')

def setup_asr(cfg):
    if not shutil.which('uv'):
        run([sys.executable,'-m','pip','install','-q','uv'],'install_uv',cfg)
    uv = shutil.which('uv')
    if not uv:
        raise RuntimeError('uv executable not found after installation')
    root = Path(cfg['work_root'])/'inference_gipformer_py311_torch280'
    python = root/'bin/python'
    if not python.is_file():
        run([uv,'python','install','3.11'],'python311',cfg)
        run([uv,'venv','--python','3.11',str(root)],'asr_venv',cfg)
    install = [uv,'pip','install','--python',str(python)]
    marker = root/'inference_ready_v1.json'
    if not marker.is_file():
        run(install+['torch==2.8.0','torchaudio==2.8.0','--index-url','https://download.pytorch.org/whl/cu128'],
            'asr_torch',cfg)
        run(install+['k2==1.24.4.dev20250807+cuda12.8.torch2.8.0',
                     '--find-links','https://k2-fsa.github.io/k2/cuda.html','--no-deps'],'asr_k2',cfg)
        run(install+['kaldifeat==1.25.5.dev20250807+cuda12.8.torch2.8.0',
                     '--find-links','https://csukuangfj.github.io/kaldifeat/cuda.html','--no-deps'],'asr_kaldifeat',cfg)
        run(install+['numpy>=1.26,<2','huggingface_hub>=0.28,<1.0','soundfile==0.13.1',
                     'sentencepiece==0.2.1','lhotse>=1.28,<2','kaldialign','kaldifst>=1.7',
                     'pypinyin==0.50.0','tensorboard','typeguard','dill','PyYAML>=6,<7',
                     'pandas>=2,<3','tqdm>=4.66','nvidia-ml-py>=12','packaging'],'asr_dependencies',cfg)
    source = Path(cfg['source_dir'])
    if not (source/'.git').is_dir():
        source.mkdir(parents=True,exist_ok=True)
        run(['git','init',str(source)],'icefall_init',cfg)
        run(['git','-C',str(source),'remote','add','origin','https://github.com/k2-fsa/icefall.git'],'icefall_remote',cfg)
    head = subprocess.run(['git','-C',str(source),'rev-parse','HEAD'],capture_output=True,text=True)
    if head.returncode != 0 or head.stdout.strip() != cfg['source_commit']:
        run(['git','-C',str(source),'fetch','--depth','1','origin',cfg['source_commit']],'icefall_fetch',cfg)
        run(['git','-C',str(source),'checkout','--detach','FETCH_HEAD'],'icefall_checkout',cfg)
    actual = subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    if actual != cfg['source_commit']:
        raise RuntimeError('Unexpected icefall checkout revision')
    probe = """import importlib.metadata as metadata
import json,torch,torchaudio,k2,kaldifeat
assert torch.cuda.is_available()
assert torch.__version__.startswith('2.8.0') and torch.version.cuda=='12.8'
def package_version(name, module):
 try:
  return metadata.version(name)
 except metadata.PackageNotFoundError:
  return str(getattr(module,'__version__',getattr(module,'__dev_version__','unknown')))
print(json.dumps({'torch':torch.__version__,'cuda':torch.version.cuda,'k2':package_version('k2',k2),
                  'kaldifeat':package_version('kaldifeat',kaldifeat),
                  'gpus':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}))
"""
    run([str(python),'-c',probe],'asr_runtime_probe',cfg)
    write_json(marker,{'source_commit':cfg['source_commit'],'torch':'2.8.0+cu128'})
    return str(python)
