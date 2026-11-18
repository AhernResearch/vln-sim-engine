"""One owned new-chain production; never replay movement or automatically restart a run."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import time
import traceback
import urllib.request
from collections import Counter


def read(path):
    try:return json.loads(path.read_text())
    except (OSError,ValueError):return None

def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.flush();os.fsync(stream.fileno())

def cache(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False));temp.replace(path)

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def ticks(pid):return Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]
def interrupted(sig,frame):raise InterruptedError(f'owned controller received signal {sig}')

def main():
    deployment=Path(__file__).parent;identity=read(deployment/'identity.json')
    root=Path(identity['remote']);plan=read(root/'production-plan.json');owner=plan['owner']
    server=deployment/'server';client_dir=root/'client'
    python='/data/nav_storage/lightnav/envs/sim-engine-031/bin/python'
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    save(root/'controller-runtime.json',{'owner':owner,'pid':os.getpid(),'start_ticks':ticks(os.getpid()),'source_sha256':sha(Path(__file__))})
    service=client=None;service_record=client_record=None;health=None
    def call(method,path,value=None,timeout=15):
        payload=None if value is None else json.dumps(value,allow_nan=False).encode()
        req=urllib.request.Request(plan['service_url']+path,data=payload,method=method,headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(req,timeout=timeout) as response:return json.load(response)
    def resources():
        mem={k:int(v.strip().split()[0])*1024 for k,v in (line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines()) if k in {'MemTotal','MemAvailable'}}
        return {'free_disk_bytes':shutil.disk_usage(root).free,'memory_bytes':mem,
                'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=10).strip().splitlines(),
                'loadavg':Path('/proc/loadavg').read_text().strip()}
    def launch(folder,label,argv,path):
        save(folder/(label+'-launch-owner.json'),{'owner':owner,'argv':argv,'automatic_restart':False,'scope':'train_free_instruction_remaining_to_100k_v5'})
        env=dict(os.environ,PYTHONPATH=path+':/data/nav_storage/nav-train/runtime/python-vendor',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
        process=subprocess.Popen(argv,cwd=folder,env=env,stdout=(folder/(label+'.log')).open('xb'),stderr=subprocess.STDOUT,start_new_session=True)
        record={'owner':owner,'pid':process.pid,'start_ticks':ticks(process.pid),'pgid':os.getpgid(process.pid)}
        save(folder/(label+'-process.json'),record)
        return process,record
    def owned_signal(folder,label,process,record,sig,group=False):
        if process.poll() is not None:return
        expected=read(folder/(label+'-launch-owner.json'))
        assert expected['owner']==record['owner']==owner and ticks(process.pid)==record['start_ticks']
        assert Path(f'/proc/{process.pid}/cmdline').read_bytes().rstrip(b'\0').decode().split('\0')==expected['argv']
        if group:
            assert os.getpgid(process.pid)==record['pgid']==process.pid
            os.killpg(record['pgid'],sig)
        else:process.send_signal(sig)
    status={'owner':owner,'client_owner':plan['client_owner'],'status':'starting','target_successes':100000,
            'credited_successes':plan['credited_successes'],'remaining_successes':plan['remaining_successes'],
            'automatic_restart':False,'automatic_retry':False,'training_exports':0,'planning_workers':32,'audit_workers':4}
    cache(root/'status.json',status)
    try:
        launch_proof=read(root/'launch-provenance.json')
        assert launch_proof['owner']==owner and sha(Path(__file__))==launch_proof['controller_sha256']
        assert sha(root/'production-plan.json')==launch_proof['production_plan_sha256']
        for relative,expected in read(server/'deployment.json')['files'].items():assert sha(server/relative)==expected,relative
        proof=read(client_dir/'source-provenance.json')
        assert sha(client_dir/'config.json')==proof['config_sha256']==plan['client_config_sha256']
        for relative,expected in proof['source_sha256'].items():assert sha(client_dir/'source'/relative)==expected,relative
        assert sha(root/'reconciliation.json')==plan['reconciliation_sha256']
        assert sha(Path(plan['old_start_pool_manifest']))==plan['old_start_pool_manifest_sha256']
        with urllib.request.urlopen('http://127.0.0.1:18812/v1/health',timeout=10) as response:mainline=json.load(response)
        assert mainline['service_owner']=='29527adbc69e4b92a8168b88bbc7874d' and mainline['active']==0
        save(root/'mainline-at-launch.json',{k:mainline[k] for k in ['service_owner','status','live_workers','active']})
        with socket.socket() as sock:assert sock.connect_ex(('127.0.0.1',18813))!=0,'production port already owned'
        for port in [18110,18111,18112,18113]:
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/v1/models',timeout=10) as response:models=json.load(response)
            assert any(m['id']=='onh-qwen27b-nvfp4' for m in models['data'])
        before=resources();save(root/'resources-at-launch.json',before)
        assert before['free_disk_bytes']>=plan['disk_reserve_bytes']+64*1024**3
        assert before['memory_bytes']['MemAvailable']>=plan['minimum_available_memory_bytes']
        service,service_record=launch(server,'service',[python,'-m','sim_engine','serve-collection','--config',str(server/'config.json'),'--host','127.0.0.1','--port','18813'],str(server/'src'))
        deadline=time.monotonic()+150
        while True:
            if service.poll() is not None:raise RuntimeError('new-chain service exited during startup; no restart')
            try:
                health=call('GET','/v1/health')
                if health['status']=='ok':break
            except OSError:pass
            if time.monotonic()>deadline:raise TimeoutError('service startup; no restart')
            time.sleep(.5)
        assert health['rolling_scene_window'] and health['workers']==32 and health['active']==0
        assert read(server/'config.json')['collection']['planning_profile_version']=='collection-v5'
        assert health['planner']['workers']==32 and health['planner']['live_workers']==32
        assert sum(s['success_goal'] for s in health['scene_schedule'])==plan['remaining_successes']
        save(root/'service-at-launch.json',health)
        status.update(service_owner=health['service_owner'],service_process=service_record)
        client,client_record=launch(client_dir,'client',[python,'-m','nav_train.pose_pipeline.short_instructions','--root',str(client_dir)],str(client_dir/'source/src'))
        status.update(status='running',client_process=client_record)
        began=time.monotonic();iteration=0;last_progress=began;last_completed=-1;health_failures=0;last_lost=set()
        with (root/'monitor.jsonl').open('x') as log:
            while True:
                try:
                    health=call('GET','/v1/health');health_failures=0;status.pop('health_error',None);status['health_failures']=0
                except OSError as error:
                    health_failures+=1
                    status.update(health_error=repr(error),health_failures=health_failures)
                    cache(root/'status.json',status)
                    if service.poll() is not None or health_failures>=3:raise RuntimeError('service health unavailable; preserve evidence')
                    time.sleep(5);continue
                status.update(evidence_io=health.get('evidence_io'),planning_io=health['planner'].get('evidence_io'))
                resource=resources();progress=read(client_dir/'progress.json')
                if progress:assert progress['owner']==plan['client_owner']
                done=0 if progress is None else progress['counts'].get('completed_audit',0)
                if done!=last_completed:last_completed=done;last_progress=time.monotonic()
                scenes=health['scene_schedule'];planner=health['planner']
                lost={w['instance'] for w in health['worker_state'] if not w['alive']}
                if lost!=last_lost:
                    save(root/'worker-events'/f'{iteration:07d}.json',{'owner':owner,'unavailable_instances':sorted(lost),'action':'continue_healthy_workers_no_replay'})
                    last_lost=lost
                status.update(elapsed_s=time.monotonic()-began,progress=progress,
                    confirmed_new_successes=done,confirmed_total_successes=plan['credited_successes']+done,
                    service_active=health['active'],live_workers=health['live_workers'],
                    server_new_successes=sum(s['successful'] for s in scenes),server_attempts=sum(s['claimed'] for s in scenes),
                    scene_status_counts=dict(Counter(s['status'] for s in scenes)),scene_schedule=scenes,
                    stages={k:health[k] for k in ['queued_observations','queued_planning','queued_executions','queued_audits']},
                    planner={'live_workers':planner['live_workers'],'workers':planner['workers'],'active':planner['active'],'queued':planner['queued']},
                    resources=resource,client_exit_code=client.poll())
                cache(root/'status.json',status);log.write(json.dumps(status)+'\n');log.flush()
                if iteration%30==0:save(root/'checkpoints'/f'{iteration:07d}.json',status)
                if client.poll() is not None:break
                if service.poll() is not None or health['live_workers']==0:raise RuntimeError('no live simulation capacity; no replay')
                if time.monotonic()-last_progress>1200:raise RuntimeError('no confirmed successful output for 1200s; inspect queues, no restart')
                if resource['free_disk_bytes']<plan['disk_reserve_bytes']:raise RuntimeError('disk reserve reached; retain all data')
                if resource['memory_bytes']['MemAvailable']<plan['minimum_available_memory_bytes']:raise RuntimeError('host memory reserve reached')
                for row in resource['gpu']:
                    gpu,used,free,util=[int(x.strip()) for x in row.split(',')]
                    if gpu in [0,2,3,4] and free<plan['gpu_reserve_mib']:raise RuntimeError('owned production GPU reserve reached')
                iteration+=1;time.sleep(10)
        summary=read(client_dir/'summary.json')
        if not summary or summary['aborted'] or summary.get('success_target_met') is not True or summary['model_threads_still_running']:
            raise RuntimeError('remaining target incomplete; frozen summary retained, no replay')
        count=summary['counts'].get('completed_audit',0)
        assert count==plan['remaining_successes'] and count+plan['credited_successes']==100000
        assert health['active']==0 and all(s['status']=='complete' and s['successful']==s['success_goal'] for s in health['scene_schedule'])
        save(root/'completion.json',{'owner':owner,'status':'completed_geometry_audit','credited_successes':plan['credited_successes'],
            'new_successes':count,'total_successes':100000,'wall_s':summary['wall_s'],
            'new_counts':summary['counts'],'semantic_admission':'pending_review','training_exports':0,'automatic_retry':False})
        status.update(status='completed_geometry_audit',confirmed_new_successes=count,confirmed_total_successes=100000)
    except BaseException as error:
        status.update(status='stopped_requires_inspection',error=repr(error))
        save(root/'controller-error.json',{'owner':owner,'error':repr(error),'traceback':traceback.format_exc(),'automatic_restart':False,'automatic_retry':False})
    finally:
        if client is not None and client.poll() is None:
            owned_signal(client_dir,'client',client,client_record,signal.SIGINT)
            try:client.wait(timeout=210)
            except subprocess.TimeoutExpired:
                owned_signal(client_dir,'client',client,client_record,signal.SIGTERM)
                try:client.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    owned_signal(client_dir,'client',client,client_record,signal.SIGKILL,group=True);client.wait(timeout=10)
        if service is not None and service.poll() is None:
            try:save(root/'service-before-stop.json',call('GET','/v1/health'))
            except OSError:pass
            owned_signal(server,'service',service,service_record,signal.SIGTERM)
            try:service.wait(timeout=60)
            except subprocess.TimeoutExpired:
                owned_signal(server,'service',service,service_record,signal.SIGKILL,group=True);service.wait(timeout=10)
        status.update(client_exit_code=client.returncode if client else None,service_exit_code=service.returncode if service else None)
        cache(root/'status.json',status);save(root/'controller-finished.json',status)


if __name__=='__main__':main()

