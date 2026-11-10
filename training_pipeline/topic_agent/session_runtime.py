"""Complete native Trainer checkpoints and portable resume and deadline signaling."""
import json
import shutil
import zipfile
from pathlib import Path, PurePosixPath
from agent_core import write_json


def checkpoint_step(path):
    return int(Path(path).name.split("-")[-1])


def checkpoint_files_valid(path, world=2, require_scaler=True):
    p=Path(path)
    required=["adapter_config.json","trainer_state.json","scheduler.pt"]
    required += [f"rng_state_{i}.pth" for i in range(world)] if world>1 else ["rng_state.pth"]
    if require_scaler:required.append("scaler.pt")
    if not any((p/n).is_file() and (p/n).stat().st_size for n in ("adapter_model.safetensors","adapter_model.bin")):return False
    if not any((p/n).is_file() and (p/n).stat().st_size for n in ("optimizer.pt","optimizer.bin")):return False
    if not all((p/n).is_file() and (p/n).stat().st_size for n in required):return False
    try:return int(json.loads((p/"trainer_state.json").read_text())["global_step"])==checkpoint_step(p)
    except (ValueError,KeyError,TypeError):return False


def complete_checkpoints(output_dir, world=2):
    paths=[]
    for p in (Path(output_dir)/"checkpoints").glob("checkpoint-*"):
        if not (p/"checkpoint_complete.json").is_file():continue
        try:record=json.loads((p/"checkpoint_complete.json").read_text())
        except ValueError:continue
        if record.get("step")==checkpoint_step(p) and checkpoint_files_valid(p,world,record.get("requires_scaler",True)):
            paths.append(p)
    return sorted(paths,key=checkpoint_step)


def commit_checkpoint(output_dir, step, epoch, world=2, requires_scaler=True, epoch_end=False, keep_latest=2):
    out=Path(output_dir);p=out/"checkpoints"/f"checkpoint-{step}"
    if not checkpoint_files_valid(p,world,requires_scaler):
        raise RuntimeError(f"Incomplete native Trainer checkpoint: {p}. Optimizer/scheduler/scaler/RNG are required for resume.")
    write_json(p/"checkpoint_complete.json",{"step":int(step),"epoch":float(epoch or 0),"requires_scaler":requires_scaler})
    manifest=out/"epoch_checkpoints.json"
    epochs=json.loads(manifest.read_text()) if manifest.is_file() else {}
    if epoch_end:
        epochs[str(int(round(epoch)))]=int(step);write_json(manifest,epochs)
    paths=complete_checkpoints(out,world)
    keep=set(epochs.values()) | {checkpoint_step(q) for q in paths[-keep_latest:]}
    for q in paths:
        if checkpoint_step(q) not in keep:shutil.rmtree(q)
    write_json(out/"latest_checkpoint.json",{"step":int(step),"checkpoint":str(p)})
    return p


def resolve_resume(cfg):
    requested=cfg.get("resume_checkpoint","auto")
    if requested in ("",None,False):return None
    if requested=="auto":
        candidates=complete_checkpoints(cfg["output_dir"],cfg["num_gpus"])
        return str(candidates[-1]) if candidates else None
    p=Path(requested)
    marker=p/"checkpoint_complete.json"
    if not marker.is_file():
        raise ValueError("Resume requires a complete checkpoint of this 3-epoch recipe")
    metadata=json.loads(marker.read_text())
    if metadata.get("step")!=checkpoint_step(p) or not checkpoint_files_valid(
            p,cfg["num_gpus"],metadata.get("requires_scaler",True)):
        raise ValueError("Resume requires a complete checkpoint of this 3-epoch recipe")
    return str(p)


def epoch_checkpoint_paths(cfg):
    out=Path(cfg["output_dir"]);manifest=out/"epoch_checkpoints.json"
    if not manifest.is_file():return []
    wanted=set(json.loads(manifest.read_text()).values())
    return [p for p in complete_checkpoints(out,cfg["num_gpus"]) if checkpoint_step(p) in wanted]


def export_resume_bundle(output_dir):
    out=Path(output_dir);target=out/"resume_state.zip";temporary=out/"resume_state.zip.tmp"
    excluded={"resume_state.zip","resume_state.zip.tmp","best_adapter.zip","reports.zip"}
    config_path=out/"run_config.json"
    world=json.loads(config_path.read_text()).get("num_gpus",2) if config_path.is_file() else 2
    complete={p.name for p in complete_checkpoints(out,world)}
    with zipfile.ZipFile(temporary,"w",compression=zipfile.ZIP_STORED,allowZip64=True) as z:
        for p in sorted(out.rglob("*")):
            relative=p.relative_to(out)
            if relative.parts[0]=="smoke_tmp":
                continue
            if relative.parts[0]=="checkpoints" and len(relative.parts)>1 and relative.parts[1] not in complete:
                continue
            if p.is_file() and p.name not in excluded and not p.name.endswith(".tmp"):
                z.write(p,PurePosixPath("run")/relative.as_posix())
    temporary.replace(target)
    return target


def restore_resume_bundle(source, output_dir):
    """Restore missing files only; never replace a newer local checkpoint."""
    src=Path(source);out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    if src.is_file():
        with zipfile.ZipFile(src) as z:
            if "run/run_config.json" not in z.namelist():raise ValueError("Expected a complete resume_state.zip")
            for item in z.infolist():
                name=PurePosixPath(item.filename)
                if name.is_absolute() or ".." in name.parts:raise ValueError("Unsafe path in resume archive")
                if not name.parts or name.parts[0]!="run" or item.is_dir():continue
                target=out.joinpath(*name.parts[1:])
                if target.exists():continue
                target.parent.mkdir(parents=True,exist_ok=True)
                with z.open(item) as r,target.open("wb") as w:shutil.copyfileobj(r,w)
    elif src.is_dir():
        if not (src/"run_config.json").is_file():src=src/"run"
        if not (src/"run_config.json").is_file():raise ValueError("Resume directory has no run_config.json")
        for p in src.rglob("*"):
            if p.is_file() and p.name!="resume_state.zip":
                target=out/p.relative_to(src)
                if not target.exists():target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    else:raise FileNotFoundError(source)
    return out


class SessionPaused(RuntimeError):
    """This Kaggle session reached its deadline and can continue from disk."""


def stop_if_deadline(cfg):
    import time
    deadline = cfg.get('soft_deadline_unix')
    if deadline is not None and time.time() >= deadline:
        raise SessionPaused('Session budget reached; resume with the saved run state.')
