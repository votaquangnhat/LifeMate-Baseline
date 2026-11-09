import os,queue,subprocess,threading,time
from pathlib import Path

def run_jobs(jobs, log_dir):
    """Stream child logs and a periodic heartbeat; stop every child on error/interrupt."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True,exist_ok=True)
    messages, workers, readers = queue.Queue(), [], []
    def reader(process,name):
        try:
            for line in process.stdout:
                messages.put((name,line))
        finally:
            messages.put((name,None))
    try:
        for name,command,env,cwd in jobs:
            handle = (log_dir / (name+'.log')).open('a',encoding='utf-8')
            handle.write('\nCOMMAND ' + repr(command) + '\n')
            handle.flush()
            process = subprocess.Popen([str(x) for x in command],env=env,cwd=cwd,stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT,text=True,bufsize=1)
            workers.append((name,process,handle))
            thread = threading.Thread(target=reader,args=(process,name),daemon=True)
            readers.append(thread)
            thread.start()
        handles = {name:handle for name,_,handle in workers}
        complete, last = set(), time.monotonic()
        while len(complete) < len(workers):
            try:
                name,line = messages.get(timeout=1)
                if line is None:
                    complete.add(name)
                else:
                    print('['+name+'] '+line,end='',flush=True)
                    handles[name].write(line)
                    handles[name].flush()
                    last = time.monotonic()
            except queue.Empty:
                if time.monotonic()-last > 30:
                    print('Still running: '+', '.join(n for n,p,_ in workers if p.poll() is None),flush=True)
                    last = time.monotonic()
            for name,process,_ in workers:
                if process.poll() not in (None,0):
                    # Drain already-captured lines before reporting the failed worker.
                    while not messages.empty():
                        n,line = messages.get_nowait()
                        if line:
                            print('['+n+'] '+line,end='',flush=True)
                            handles[n].write(line)
                    raise RuntimeError(f'{name} failed (exit {process.returncode}). See {log_dir/(name+".log")}')
        for name,process,_ in workers:
            if process.wait() != 0:
                raise RuntimeError(name+' failed; inspect its log')
    finally:
        for _,process,_ in workers:
            if process.poll() is None:
                process.terminate()
        for _,process,handle in workers:
            if process.poll() is None:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            handle.close()
        for thread in readers:
            thread.join(timeout=2)
        for _,process,_ in workers:
            process.stdout.close()
