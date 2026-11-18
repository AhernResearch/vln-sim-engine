"""Launch exactly one frozen continuation with generated process ownership."""
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def save(p, value):
    with p.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())


def main():
    deployment = Path(__file__).parent
    state = read(deployment / 'identity.json')
    root = Path(state['remote'])
    plan = read(root / 'production-plan.json')
    assert plan['owner'] == state['owner']
    assert plan['planning_workers'] == 32
    assert plan['workers'] == 32 and plan['planning_profile_version'] == 'collection-v5'
    assert plan['credited_successes'] + plan['remaining_successes'] == 100000
    assert resource.getrlimit(resource.RLIMIT_FSIZE) == (-1, -1)
    # The deployed binaries/sources must be the exact optimized E2E-tested snapshot.
    e2e = Path('/data/nav_storage/nav-train/collections/planning-e2e-1e6e8f1d82c8')
    proof = read(e2e / 'source-manifest.json')
    matches = {}
    for variant, location in [('optimized', deployment / 'server'), ('client', root / 'client/source')]:
        expected = {name[len(variant) + 1:]: digest for name, digest in proof.items() if name.startswith(variant + '/')}
        actual = {str(p.relative_to(location)): sha(p) for p in (location / 'src').rglob('*') if p.is_file()}
        if variant == 'optimized':
            assert set(actual) - set(expected) == {'src/sim_engine/async_io.py'}
            assert not set(expected) - set(actual)
            changed = {n for n in expected if expected[n] != actual[n]}
            assert changed == {'src/sim_engine/collection/server.py', 'src/sim_engine/planning/service.py', 'src/sim_engine/planning/client.py'}, changed
            matches[variant] = {'unchanged_files':len(expected)-len(changed), 'changed':sorted(changed), 'added':['src/sim_engine/async_io.py']}
        else:
            assert set(expected) == set(actual), variant
            changed = [n for n in expected if expected[n] != actual[n]]
            assert changed == ['src/nav_train/collection_client.py'], changed
            matches[variant] = {'unchanged_files': len(expected)-1, 'changes': {n: {'before':expected[n], 'after':actual[n]} for n in changed}}
    validation = read(deployment / 'validation.json')
    assert validation['passed'] is True
    remote_validation=read(deployment / 'remote-validation.json')
    assert remote_validation['passed'] is True
    for name,digest in validation['server_sources'].items():assert sha(deployment / 'server/src/sim_engine' / name)==digest
    assert validation['collection_client_sha256'] == sha(root / 'client/source/src/nav_train/collection_client.py')
    save(root / 'handoff-validation.json', {
        'owner': state['owner'], 'source_match_e2e': matches,
        'e2e_root': str(e2e), 'e2e_manifest_sha256': sha(e2e / 'source-manifest.json'),
        'e2e_result_sha256': sha(e2e / 'result.json'),
        'previous_finished_sha256': sha(Path(state['previous_root']) / 'controller-finished.json'),
        'file_size_limit': list(resource.getrlimit(resource.RLIMIT_FSIZE)),
        'automatic_retry': False, 'automatic_restart': False})
    controller = deployment / 'supervise_free_v5_planning32_production.py'
    save(root / 'launch-provenance.json', {
        'owner': state['owner'], 'controller_sha256': sha(controller),
        'preparation_sha256': sha(deployment / 'prepare_free_v5_production.py'),
        'launcher_sha256': sha(Path(__file__)),
        'production_plan_file': 'production-plan.json',
        'continuation_adjustments_sha256': sha(root / 'continuation-adjustments.json'),
        'production_plan_sha256': sha(root / 'production-plan.json'),
        'handoff_validation_sha256': sha(root / 'handoff-validation.json'),
        'frozen_source_sha256': sha(deployment / 'frozen-source.json'),
        'automatic_retry': False, 'automatic_restart': False,
        'formal_benchmark': False, 'training_exports': 0})
    argv = ['/data/nav_storage/lightnav/envs/sim-engine-031/bin/python', str(controller)]
    save(root / 'controller-launch-owner.json', {'owner': state['owner'], 'argv': argv, 'automatic_restart': False})
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    with (root / 'controller.log').open('xb') as log:
        process = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                   cwd=deployment, env=env, start_new_session=True)
    record = {'owner': state['owner'], 'pid': process.pid,
              'start_ticks': Path('/proc', str(process.pid), 'stat').read_text().rsplit(')', 1)[1].split()[19],
              'pgid': os.getpgid(process.pid), 'argv': argv}
    save(root / 'controller-process.json', record)
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()

