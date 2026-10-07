"""Pure routing/synthetic tests; synthetic seed7 only, no canonical RNG/data."""
import copy
import types

import pytest
from copeland_erdos_nets.attempt_transport import namespace, validate_manifest, existing_target_id, SUFFIXES, CONFIG_SHA256

def manifest(attempt='fixture2', offset=0):
    prefix=namespace(7,attempt)
    return {'version':1,'seed':7,'attempt_id':attempt,'namespace':prefix,'source_sha':'a'*40,
            'config_sha256':CONFIG_SHA256,'exchange_id':'same-exchange',
            'sa_principal':'same-sa','ds_authority_path':'canonical-review.md','ds_authority_sha256':'b'*64,
            'forbidden_ids':['legacy-id'],
            'targets':{s:{'name':prefix+s,'id':f'id-{i+offset}','parent_id':'same-exchange'} for i,s in enumerate(SUFFIXES)}}

def test_two_attempts_complete_and_disjoint():
    a,b=manifest('fixture1'),manifest('fixture2',100)
    for m in (a,b):validate_manifest(m,7,m['attempt_id'],'a'*40)
    assert len(a['targets'])==len(b['targets'])==14
    assert not {t['id'] for t in a['targets'].values()} & {t['id'] for t in b['targets'].values()}
    assert not {t['name'] for t in a['targets'].values()} & {t['name'] for t in b['targets'].values()}

@pytest.mark.parametrize('attempt',['','../bad','a/b','a\\b','.','a b','-start','a'*97])
def test_invalid_attempt_rejected(attempt):
    with pytest.raises(ValueError):namespace(7,attempt)

@pytest.mark.parametrize('kind',['missing','duplicate','parent','namespace','forbidden','source','seed','config'])
def test_manifest_fail_closed(kind):
    m=manifest()
    if kind=='missing':del m['targets']['_state.json']
    if kind=='duplicate':m['targets']['_state.json']['id']=m['targets']['_live.csv']['id']
    if kind=='parent':m['targets']['_state.json']['parent_id']='other-exchange'
    if kind=='namespace':m['namespace']='gate_bc_seed_7'
    if kind=='forbidden':m['targets']['_state.json']['id']='legacy-id'
    if kind=='source':m['source_sha']='c'*40
    if kind=='seed':m['seed']=8
    if kind=='config':m['config_sha256']='0'*64
    with pytest.raises(ValueError):validate_manifest(m,7,'fixture2','a'*40)

@pytest.mark.parametrize('case',['missing','duplicate','wrong-id'])
def test_lookup_never_falls_back(case):
    m=manifest()
    records={'missing':[],'duplicate':[{'id':'id-0'},{'id':'id-0'}],'wrong-id':[{'id':'legacy-id'}]}[case]
    drive=types.SimpleNamespace(ListFile=lambda query:types.SimpleNamespace(GetList=lambda:records))
    with pytest.raises(ValueError):existing_target_id(drive,m,'_state.json')

def test_all_writer_targets_exact_existing_ids():
    m=manifest()
    seen=[]
    def listing(query):
        assert "'same-exchange' in parents" in query['q']
        suffix=next(s for s in SUFFIXES if "title='"+m['namespace']+s+"'" in query['q'])
        seen.append(suffix)
        return types.SimpleNamespace(GetList=lambda:[{'id':m['targets'][suffix]['id']}])
    drive=types.SimpleNamespace(ListFile=listing)
    for suffix in SUFFIXES:assert existing_target_id(drive,m,suffix)==m['targets'][suffix]['id']
    assert set(seen)==set(SUFFIXES) # state/failure sync, checkpoints, archive/readback receipt
    with pytest.raises(ValueError):existing_target_id(drive,m,'legacy_state.json')

def test_source_writers_share_manifest():
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    source=(root/'scripts/run_gate_bc_seed_atomic.py').read_text()
    assert 'no legacy fallback' in source
    assert "DriveUpdates(out, transport_manifest['namespace'], transport_manifest)" in source
    assert 'drive.upload(out/"state.json", "_state.json")' in source
    assert 'update_state("EVIDENCE_INCOMPLETE"' in source

@pytest.mark.parametrize('expiry',['before_update','between_update_readback','failure_status'])
def test_mapped_update_expiry_and_hash(transport, expiry):
    # Reuse the refresh fixture to exercise the real DriveUpdates.upload path.
    credentials,auth,updater,payload,refreshes,actions=transport
    m=manifest()
    m['targets']['_state.json']['id']='existing-target-id'
    updater.manifest=m
    updater.prefix=m['namespace']
    credentials.expire_after_update=expiry=='between_update_readback'
    credentials.access_token_expired=expiry in ('before_update','failure_status')
    updater.upload(payload,'_state.json')
    assert actions==['update','readback'] and len(refreshes)==1
    assert updater.receipts[-1]['readback_pass']

@pytest.mark.parametrize('failure',['refresh','permission','corrupt'])
def test_mapped_failure_no_fallback_or_retry(transport, failure):
    credentials,auth,updater,payload,refreshes,actions=transport
    m=manifest()
    m['targets']['_state.json']['id']='existing-target-id'
    updater.manifest=m
    updater.prefix=m['namespace']
    if failure=='refresh':credentials.access_token_expired=credentials.fail_refresh=True
    if failure=='permission':credentials.permission_error=True
    if failure=='corrupt':credentials.corrupt=True
    with pytest.raises((RuntimeError,PermissionError)):updater.upload(payload,'_state.json')
    assert len(refreshes)<=1 and actions.count('update')<=1 and actions.count('readback')<=1

# Fixture imported rather than invoking any seed-specific scientific runner.
from test_sa_drive_refresh import transport

def test_main_science_ast_unchanged_except_isolated_transport_hook():
    import ast
    from pathlib import Path
    import subprocess
    root=Path(__file__).resolve().parents[1]
    path='scripts/run_gate_bc_seed_atomic.py'
    old=ast.parse(subprocess.check_output(['git','show','c892bf0aea62e9e8c7419311e810e8ad09491811:'+path],cwd=root))
    new=ast.parse((root/path).read_text())
    a=next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    b=next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    remove=[]
    for n in b.body:
        if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute) and n.value.func.attr=='add_argument' and n.value.args and isinstance(n.value.args[0],ast.Constant) and n.value.args[0].value=='--transport-manifest':remove.append(n)
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='transport_manifest' for t in n.targets):remove.append(n)
        if isinstance(n,ast.If) and 'args.mode' in ast.unparse(n.test) and 'GATE_BC_DRIVE_LIVE' in ast.unparse(n.test):remove.append(n)
    assert len(remove)==3
    b.body=[n for n in b.body if n not in remove]
    old_try=next(n for n in a.body if isinstance(n,ast.Try))
    new_try=next(n for n in b.body if isinstance(n,ast.Try))
    assert isinstance(new_try.body[0],ast.If)
    assert 'DriveUpdates' in ast.unparse(new_try.body[0]) and 'transport_manifest' in ast.unparse(new_try.body[0])
    new_try.body[0]=copy.deepcopy(old_try.body[0])
    assert ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False)
