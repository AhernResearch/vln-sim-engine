import json
import zipfile
from pathlib import Path
from sim_engine.scan_assets import prepare, sha256, write_json, ply_header

root=Path(__file__).resolve().parent
rows=[]
for name,scene,lod in [('智元-output','zhiyuan',3),('智园1-output','zhiyuan1',4),('智园2-output','zhiyuan2',3)]:
    source=Path('/home/holo')/name
    relative=[f'ply-result/point_cloud/iteration_100/point_cloud_{lod}.ply','mesh/mesh.ply','render/assets/poses.json']
    archive=root/(scene+'-selected.zip')
    source_files={r:{'sha256':sha256(source/r),'bytes':(source/r).stat().st_size} for r in relative}
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_STORED) as bundle:
        for r in relative: bundle.write(source/r,name+'/'+r)
    manifest=prepare(archive,root/'scenes'/scene,lod)
    data=json.loads(manifest.read_text())
    data['scene_id']=scene
    data['source']['directory']=str(source)
    data['source']['selected_files']=source_files
    manifest.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    row={'name':name,'scene_id':scene,'lod':lod,'manifest':str(manifest),'gaussians':data['assets']['render']['vertices'],'mesh_vertices':data['assets']['mesh']['vertices'],'scan_positions':len(json.loads((manifest.parent/'scan_positions.json').read_text())['positions']),'source_files':source_files}
    rows.append(row)
    print(json.dumps(row,ensure_ascii=False),flush=True)
write_json(root/'inputs.json',rows)
