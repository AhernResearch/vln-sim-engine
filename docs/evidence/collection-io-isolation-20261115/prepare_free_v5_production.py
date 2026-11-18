"""Freeze a new v5 continuation; immutable prior attempts are never replayed."""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
from collections import Counter
from itertools import zip_longest


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())


def key(row):
    ep = row.get('episode', row)
    return ep['scene_id'], tuple(ep['start_position']), tuple(ep['start_rotation'])


def main():
    deployment = Path(__file__).parent
    state = read(deployment / 'identity.json')
    root = Path(state['remote'])
    root.mkdir(parents=True, exist_ok=False)
    save(root / 'ownership.json', state)
    previous = Path(state['previous_root'])
    oldplan = read(previous / 'production-plan.json')
    finished = read(previous / 'controller-finished.json')
    summary = read(previous / 'client/summary.json')
    assert finished['owner'] == oldplan['owner']
    assert summary['model_threads_still_running'] == 0
    for filename in ['controller-runtime.json', 'client/client-process.json', 'controller-process.json']:
        process = read(previous / filename)
        proc = Path('/proc', str(process['pid']), 'stat')
        assert not proc.exists() or proc.read_text().rsplit(')', 1)[1].split()[19] != process['start_ticks']
    oldclient = previous / 'client'
    oldproof = read(oldclient / 'source-provenance.json')
    assert sha(oldclient / 'config.json') == oldproof['config_sha256']
    for relative, expected in oldproof['source_sha256'].items():
        assert sha(oldclient / 'source' / relative) == expected
    config = read(oldclient / 'config.json')
    oldrecon = read(previous / 'reconciliation.json')
    assert sha(previous / 'reconciliation.json') == oldplan['reconciliation_sha256']
    credited = list(oldrecon['records'])
    for record in credited:
        folder = Path(record['client_folder'])
        rawfolder = Path(record['raw_folder'])
        for path, name in [(folder / 'result.json', 'result_sha256'),
                           (rawfolder / 'receipt.json', 'receipt_sha256'),
                           (folder / 'pointnav-target.json', 'pointnav_target_sha256'),
                           (folder / 'pointnav-outcome.json', 'pointnav_outcome_sha256')]:
            assert sha(path) == record[name], str(path)
    inherited = len(credited)
    raw = Path(oldplan['shared_root']) / 'tasks'
    raw_requests = {p.parent.name: read(p) for p in raw.glob('*/request.json')}
    blocked = {key(value) for value in raw_requests.values()}
    # Also exclude observation-only / smoke / paired experiment poses, even if they
    # happen to occur in the unclaimed frozen production pool.
    diagnostic_requests = 0
    patterns = [('/data/nav_storage/sim-engine/deployments', 'planning-*/**/request.json'),
                ('/data/nav_storage/nav-train/collections', 'planning-e2e-*/phase-*/collection/tasks/*/request.json')]
    for parent, pattern in patterns:
        for path in Path(parent).glob(pattern):
            item = read(path)
            if item.get('kind') == 'short_instructions' and 'start_position' in item:
                blocked.add(key(item))
                diagnostic_requests += 1
    claims = {}
    for entry in os.scandir(oldclient):
        if not re.fullmatch(r's[0-9]+-p[0-9]+', entry.name) or not entry.is_dir(follow_symlinks=False):
            continue
        lease = Path(entry.path) / 'lease.json'
        if lease.is_file():
            record = read(lease)
            assert record['owner'] == config['owner']
            claims[entry.name] = record['request_id']
    new_credit_scenes = Counter()
    uncredited = []
    found = set()
    eligible = []
    print(json.dumps({'phase': 'reconciling', 'inherited_credit': inherited,
                      'raw_claims': len(raw_requests), 'client_claims': len(claims),
                      'diagnostic_requests': diagnostic_requests}), flush=True)
    for case in config['cases']:
        if case['id'] in claims:
            cid = claims[case['id']]
            assert cid == hashlib.sha256((config['owner'] + ':' + case['id']).encode()).hexdigest()
            found.add(case['id'])
            blocked.add(key(case))
            folder = oldclient / case['id']
            result_path = folder / 'result.json'
            receipt_path = raw / cid / 'receipt.json'
            result = read(result_path) if result_path.exists() else None
            receipt = read(receipt_path) if receipt_path.exists() else None
            if result and result.get('status') == 'completed_audit':
                assert receipt and receipt['status'] == 'completed_audit', case['id']
                target = read(folder / 'pointnav-target.json')
                outcome = read(folder / 'pointnav-outcome.json')
                selected = read(folder / 'selected-goal.json')
                prepared = read(folder / 'private-geometry.json')
                recording = read(folder / 'recording-check.json')
                terminal = read(folder / 'terminal.json')
                assert target['owner'] == config['owner'] and target['source_split'] == 'train'
                goal = target['episode']['goals'][0]['position']
                assert goal == selected['position'] == prepared['requested_goal_position'] == outcome['intended_goal_position']
                assert selected['source'] == 'planned_before_movement'
                assert outcome['status'] == 'completed_audit' and outcome['collision_steps'] == 0 and outcome['inside_original_bbox']
                assert result['geometry']['passed'] and result['recording_complete'] and receipt['recording_complete']
                assert terminal['receipt_stop'] and terminal['frame'] + 1 == recording['frames']
                assert recording['conversion_inputs_complete'] and recording['control_dt_s'] == .25
                assert outcome['final_geodesic_distance_m'] <= target['success_criteria']['goal_tolerance_m']
                frames = 0
                with (folder / 'world_states.jsonl').open() as a, (folder / 'pointnav-frames.jsonl').open() as b:
                    for aa, bb in zip_longest(a, b):
                        assert aa is not None and bb is not None, case['id']
                        aa, bb = json.loads(aa), json.loads(bb)
                        assert aa['frame'] == bb['frame'] == frames
                        assert aa['control_time_s'] == bb['control_time_s']
                        assert all(math.isfinite(x) for x in bb['goal_base_forward_left_up_m'])
                        frames += 1
                assert frames == recording['frames'], case['id']
                credited.append({'id': case['id'], 'request_id': cid, 'scene_id': case['episode']['scene_id'],
                                 'client_folder': str(folder), 'raw_folder': str(raw / cid), 'frames': frames,
                                 'result_sha256': sha(result_path), 'receipt_sha256': sha(receipt_path),
                                 'pointnav_target_sha256': sha(folder / 'pointnav-target.json'),
                                 'pointnav_outcome_sha256': sha(folder / 'pointnav-outcome.json')})
                new_credit_scenes[case['episode']['scene_id']] += 1
                if (len(credited) - inherited) % 2000 == 0:
                    print(json.dumps({'phase': 'verified', 'new_credit': len(credited) - inherited}), flush=True)
            elif receipt and receipt.get('status') == 'completed_audit':
                uncredited.append({'id': case['id'], 'request_id': cid,
                                   'reason': 'client_PointNav_completion_not_confirmed'})
        if key(case) not in blocked:
            eligible.append(case)
    assert found == set(claims)
    assert len(credited) - inherited == summary['counts'].get('completed_audit', 0)
    assert len({(r['client_folder'], r['request_id']) for r in credited}) == len(credited)
    available = Counter(case['episode']['scene_id'] for case in eligible)
    schedule = []
    fulfilled_scenes = []
    for row in oldplan['scene_schedule']:
        goal = row['success_goal'] - new_credit_scenes[row['scene_id']]
        assert goal >= 0
        if goal == 0:
            fulfilled_scenes.append(row['scene_id'])
            continue
        attempts = min(3 * goal, available[row['scene_id']])
        assert attempts >= goal, (row['scene_id'], attempts, goal)
        schedule.append(dict(row, success_goal=goal, attempts=attempts))
    schedule.sort(key=lambda row: (-row['success_goal'], row['scene_id']))
    budget = {row['scene_id']: row['attempts'] for row in schedule}
    by_scene = {scene: [] for scene in budget}
    for case in eligible:
        scene = case['episode']['scene_id']
        if scene in budget and len(by_scene[scene]) < budget[scene]:
            by_scene[scene].append(case)
    cases = [case for row in schedule for case in by_scene[row['scene_id']]]
    assert len({case['id'] for case in cases}) == len(cases)
    assert all(key(case) not in blocked for case in cases)
    remaining = sum(row['success_goal'] for row in schedule)
    assert remaining + len(credited) == 100000
    reconciliation = {'owner': state['owner'], 'previous_root': str(previous),
                      'previous_owner': oldplan['owner'], 'previous_finished_sha256': sha(previous / 'controller-finished.json'),
                      'previous_reconciliation_sha256': sha(previous / 'reconciliation.json'),
                      'inherited_completed_PointNav': inherited,
                      'previous_run_completed_PointNav': len(credited) - inherited,
                      'credited_completed_PointNav': len(credited), 'remaining_successes': remaining,
                      'previous_raw_claims': len(raw_requests), 'previous_client_leases': len(claims),
                      'diagnostic_requests': diagnostic_requests, 'excluded_unique_poses': len(blocked),
                      'uncredited_server_completions': uncredited, 'fulfilled_scenes': fulfilled_scenes,
                      'scene_credits': dict(Counter(r['scene_id'] for r in credited)), 'records': credited,
                      'automatic_retry': False, 'training_exports': 0,
                      'checks': 'inherited result/receipt/PointNav hashes; all new complete receipts, frozen goals, 4Hz and per-frame finite aligned labels'}
    save(root / 'reconciliation.json', reconciliation)
    client = root / 'client'
    client.mkdir()
    shutil.copytree(deployment / 'client/source', client / 'source')
    base = {k: v for k, v in config.items() if k != 'cases'}
    base.update(owner=state['client_owner'], purpose='free_instruction_100k_remaining_v5', compact_request_evidence=True)
    base['collection_service'].update(url=state['service_url'], shared_root=state['shared_root'], max_pending=128, per_scene_pending=8, http_timeout_s=120)
    save(client / 'base-config.json', base)
    save(client / 'config.json', dict(base, cases=cases))
    save(client / 'source-provenance.json', {
        'owner': state['client_owner'], 'source_sha256': {str(p.relative_to(client / 'source')): sha(p) for p in (client / 'source').rglob('*') if p.is_file()},
        'config_sha256': sha(client / 'config.json'), 'reconciliation_sha256': sha(root / 'reconciliation.json'),
        'start_pool_manifest_sha256': oldplan['old_start_pool_manifest_sha256'],
        'entrypoint_module': 'nav_train.pose_pipeline.short_instructions', 'status': 'prepared_not_launched',
        'model_requests': 0, 'movement_commands': 0, 'training_exports': 0})
    settings = read(Path(state['previous_deployment']) / 'server/config.json')
    settings['collection']['planning_workers'] = 32
    worker_gpus = []
    gpu_usage = Counter()
    scenes_root = Path(settings['simulation']['scenes_root'])
    for scene in schedule[:32]:
        occupied = Counter(worker_gpus)
        gpu = min([g for g in [0, 2, 3, 4] if occupied[g] < 8], key=lambda g: (gpu_usage[g], g))
        worker_gpus.append(gpu)
        gpu_usage[gpu] += (scenes_root / scene['scene_id']).stat().st_size
    settings['collection'].update(root=state['shared_root'], worker_gpus=worker_gpus,
                                  scene_schedule=schedule, planning_profile_version='collection-v5')
    from sim_engine.config import Settings
    from sim_engine.planning.contracts import PlanningProfile
    from sim_engine.planning.native import require_native
    # Fail before any model request/movement if the mandatory v5 ABI is missing.
    require_native()
    assert PlanningProfile().version == 'collection-v5'
    settings = Settings.model_validate(settings).model_dump(mode='json')
    server = deployment / 'server'
    save(server / 'config.json', settings)
    save(server / 'deployment.json', {'owner': state['owner'], 'files': {str(p.relative_to(server)): sha(p) for p in server.rglob('*') if p.is_file()}})
    plan = dict(state, target_successes=100000, credited_successes=len(credited), remaining_successes=remaining,
                maximum_new_attempts=len(cases), scenes=len(schedule), total_training_scenes=61,
                scene_schedule=schedule, workers=32, planning_workers=32, audit_workers=4,
                planning_capacity=128, model_slots=64, prefetch=128, per_scene_pending=8,
                worker_gpus=worker_gpus, planning_budget=oldplan['planning_budget'],
                disk_reserve_bytes=oldplan['disk_reserve_bytes'], gpu_reserve_mib=6144,
                minimum_available_memory_bytes=oldplan['minimum_available_memory_bytes'],
                reconciliation_sha256=sha(root / 'reconciliation.json'),
                client_config_sha256=sha(client / 'config.json'), server_config_sha256=sha(server / 'config.json'),
                old_start_pool_manifest=oldplan['old_start_pool_manifest'],
                old_start_pool_manifest_sha256=oldplan['old_start_pool_manifest_sha256'],
                automatic_restart=False, automatic_retry=False, training_exports=0, formal_benchmark=False,
                admission='audit_only_pending_semantic_review', state='prepared_not_launched')
    save(root / 'production-plan.json', plan)
    save(root / 'continuation-adjustments.json', {
        'owner': state['owner'], 'simulation_workers': 32, 'planning_workers': 32,
        'previous_http_timeout_s': 30, 'http_timeout_s': 120,
        'automatic_retry': False, 'automatic_restart': False,
        'reason': 'isolate blocking evidence I/O from HTTP loop; preserve durable transitions, no movement replay'})
    print(json.dumps({k: v for k, v in plan.items() if k not in {'scene_schedule', 'worker_gpus'}}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()

