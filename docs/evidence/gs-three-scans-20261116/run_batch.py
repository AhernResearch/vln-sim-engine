import json,os,subprocess,time,traceback
from pathlib import Path
root=Path(__file__).resolve().parent
runtime=Path('/data/nav_storage/open-nav-harness-tools/habitat-gs/3066ae942703477c859921c1a94c55c4')
deploy=root
python=str(runtime/'venv/bin/python')
env=dict(os.environ,PYTHONPATH=f'{deploy}/src:{deploy}/deps',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',HABITAT_SIM_LOG='quiet',MAGNUM_LOG='quiet')
rows=[]
for inp in json.loads((root/'inputs.json').read_text()):
    scene=inp['scene_id']; folder=root/'scenes'/scene
    row={**inp,'manifest':str(folder/'scene.json'),'stages':{},'started_at':time.time()}
    rows.append(row)
    config={'simulation':{'backend':'habitat_gs','sensors':'rgbd','scenes_root':str(folder),'gpu':2,'workers':1,'datasets':{scene:{'episodes':str(folder/'navigation/episodes.json'),'scenes':{scene:str(folder/'scene.json')},'task':'development','physics':{'agent_radius_m':0.18,'agent_height_m':1.5}}}},'policy':{'model':'/unused'},'cameras':{'gs_probe':{'width':480,'height':270,'height_m':0.88,'pitch_down_rad':0,'hfov':120}}}
    config_path=root/(scene+'.json'); config_path.write_text(json.dumps(config,indent=2))
    commands={'bake':[python,'-m','sim_engine','bake-scan','--manifest',str(folder/'scene.json')], 'probe':[python,str(root/'probe_scan.py'),'--config',str(config_path),'--output',str(root/(scene+'-probe'))]}
    for stage,command in commands.items():
        print(scene,stage,flush=True)
        try:
            with (root/(scene+'-'+stage+'.log')).open('w') as log:
                done=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=360)
            row['stages'][stage]={'exit_code':done.returncode}
            if done.returncode: break
        except Exception as exc:
            row['stages'][stage]={'error':repr(exc)}; break
        finally:
            (root/'summary.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
    row['finished_at']=time.time()
    (root/'summary.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
print('FINISHED',flush=True)
