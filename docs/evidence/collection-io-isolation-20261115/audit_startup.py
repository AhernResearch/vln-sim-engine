"""Read-only startup sample verification; writes a separate run-level audit only."""
import json
import math
from pathlib import Path
import os
from collections import Counter

import numpy as np
from sim_engine.collection.recording import validate_recording
from sim_engine.planning.contracts import fingerprint
from nav_train.pose_pipeline.pointnav import frame_goal
from nav_train.request_evidence import restore_request


def read(path):return json.loads(path.read_text())

def main():
    deployment=Path(__file__).parent;identity=read(deployment/'identity.json');root=Path(identity['remote'])
    rows=[];failures=[]
    for path in (root/'client').glob('*/result.json'):
        result=read(path)
        if result['status']=='completed_audit' and len(rows)<5:rows.append((path.parent,result))
        elif result['status']=='failed_or_uncertain' and len(failures)<8:
            source=Path(identity['shared_root'])/'tasks'/result['service_request_id']
            receipt=read(source/'receipt.json');failures.append({k:receipt.get(k) for k in ['request_id','phase','error','movement_attempted','planning_outcome']})
    assert len(rows)==5,'wait for five completed records before auditing'
    checked=[]
    for folder,result in rows:
        cid=result['service_request_id'];raw=Path(identity['shared_root'])/'tasks'/cid
        request=read(raw/'request.json');receipt=read(raw/'receipt.json')
        target=read(folder/'pointnav-target.json');outcome=read(folder/'pointnav-outcome.json')
        prepared=read(raw/'private-geometry.json');selected=read(raw/'selected-goal.json')
        claim=read(raw/'planning-claim.json');plan=read(raw/'planning-result.json');lease=read(raw/'execution-lease.json')
        movement=read(raw/'movement-attempted.json')
        assert receipt['status']=='completed_audit' and target['source_split']=='train'
        assert target['owner']==identity['client_owner'] and target['episode']['start_position']==request['start_position']
        goal=target['episode']['goals'][0]['position']
        assert goal==selected['position']==prepared['requested_goal_position']==outcome['intended_goal_position']
        assert plan['status']=='feasible' and claim['input_digest']==plan['input_digest']
        assert plan['result_digest']==fingerprint({k:v for k,v in plan.items() if k!='result_digest'})
        assert lease['planning_attempt']==claim['attempt_id']==plan['attempt_id']
        assert lease['plan_digest']==movement['plan_digest']==plan['result_digest']
        assert lease['identity']==movement['execution_lease'] and not movement['automatic_retry']
        recording=validate_recording(raw)
        assert recording['conversion_inputs_complete'] and recording['control_dt_s']==.25
        states=[json.loads(line) for line in (raw/'world_states.jsonl').read_text().splitlines()]
        labels=[json.loads(line) for line in (folder/'pointnav-frames.jsonl').read_text().splitlines()]
        assert len(states)==len(labels)==recording['frames']
        for state,label in zip(states,labels):
            expected=frame_goal(state,goal)
            for name in ['goal_base_forward_left_up_m','goal_distance_planar_m','goal_bearing_left_rad']:
                np.testing.assert_allclose(label[name],expected[name],atol=1e-7,rtol=0)
        assert result['geometry']['collision_steps']==0 and result['geometry']['final_distance_m']<=.12
        assert receipt['region_terminal']['inside_original_bbox'] and outcome['status']=='completed_audit'
        public=read(raw/'public-observation.json');assert set(public)=={'images'}
        wire=read(folder/'wire-request.json');restored=restore_request(wire)
        images=[part for message in restored['messages'] if isinstance(message['content'],list) for part in message['content'] if part.get('type')=='image_url']
        assert len(images)==4
        checked.append({'id':folder.name,'request_id':cid,'frames':len(states),'final_distance_m':result['geometry']['final_distance_m'],
            'inside_original_bbox':True,'collision_steps':0,'recording_valid':True,'per_frame_PointNav_valid':True,
            'plan_digest_verified':True,'execution_lease_verified':True,'model_input_images_restored':4})
    report={'owner':identity['owner'],'status':'passed','sample_size':len(checked), 'audit_source_sha256': __import__('hashlib').sha256(Path(__file__).read_bytes()).hexdigest(),'records':checked,
        'sample_frames':sum(row['frames'] for row in checked),'early_uncertain_receipts':failures,
        'semantic_review':'pending_review','automatic_retry':False,'training_exports':0}
    with (root/'launch-audit.json').open('x') as stream:
        json.dump(report,stream,indent=2,ensure_ascii=False);stream.flush();os.fsync(stream.fileno())
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':main()

