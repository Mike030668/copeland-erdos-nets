"""Forced-expiry transport checks; no data/model/canonical RNG activity."""
import ast
import hashlib
import importlib.util
from pathlib import Path
import subprocess
import types

import pytest
from pydrive2.auth import LoadAuth
from copeland_erdos_nets.sa_drive_auth import ServiceAccountOnlyAuth, ServiceAccountCredentials

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def transport(tmp_path, monkeypatch):
    credentials = types.SimpleNamespace(service_account_email='same-sa@example.iam.gserviceaccount.com',
        access_token_expired=False, refresh_token=None, set_store=lambda store: None)
    monkeypatch.setattr(ServiceAccountCredentials, 'from_json_keyfile_name', lambda *a: credentials)
    auth = ServiceAccountOnlyAuth('/unchanged/sa.json')
    auth.service = object()
    auth.thread_local.http = object()
    refreshes = []
    def refresh(self):
        refreshes.append(self.credentials.service_account_email)
        if getattr(credentials,'fail_refresh',False): raise RuntimeError('refresh failed')
        credentials.access_token_expired = False
    monkeypatch.setattr(ServiceAccountOnlyAuth,'Refresh',refresh)
    stored, actions = {}, []
    class File:
        def __init__(self, metadata):
            assert metadata == {'id':'existing-target-id'}
            self.auth = auth
            self.path = None
        def SetContentFile(self, path): self.path = Path(path)
        @LoadAuth
        def Upload(self):
            actions.append('update')
            if getattr(credentials,'permission_error',False): raise PermissionError('403')
            stored['bytes'] = self.path.read_bytes()
            if getattr(credentials,'expire_after_update',False): credentials.access_token_expired = True
        @LoadAuth
        def GetContentFile(self, path):
            actions.append('readback')
            Path(path).write_bytes(stored['bytes'] + (b'corrupt' if getattr(credentials,'corrupt',False) else b''))
    class Drive:
        def ListFile(self, query):
            assert 'trashed=false' in query['q']
            return types.SimpleNamespace(GetList=lambda:[{'id':'existing-target-id'}])
        def CreateFile(self, metadata): return File(metadata)
    spec = importlib.util.spec_from_file_location('sa_refresh_runner',ROOT/'scripts/run_gate_bc_seed_atomic.py')
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    updater = runner.DriveUpdates.__new__(runner.DriveUpdates)
    updater.auth, updater.drive, updater.out = auth, Drive(), tmp_path
    updater.prefix, updater.parent, updater.receipts = 'unchanged-prefix', 'unchanged-exchange', []
    payload = tmp_path/'payload.json'
    payload.write_bytes(b'{"scope":"NONSCIENTIFIC"}\n')
    return credentials,auth,updater,payload,refreshes,actions


@pytest.mark.parametrize('expiry', ['valid','before_update','between_update_readback','failure_status'])
def test_update_readback_expiry(transport, expiry):
    credentials,auth,updater,payload,refreshes,actions = transport
    credentials.access_token_expired = expiry in ('before_update','failure_status')
    credentials.expire_after_update = expiry == 'between_update_readback'
    updater.upload(payload, '_state.json' if expiry=='failure_status' else '_checkpoint.pt')
    assert actions == ['update','readback']
    assert len(refreshes) == (0 if expiry=='valid' else 1)
    assert all(p == auth.principal for p in refreshes)
    assert updater.receipts[0]['readback_pass']
    assert updater.receipts[0]['sha256'] == hashlib.sha256(payload.read_bytes()).hexdigest()
    assert all(e['principal_before']==e['principal_after']==auth.principal and not e['user_oauth_invoked'] for e in auth.refresh_events)


def test_refresh_failure_hard_fails_no_retry(transport):
    credentials,auth,updater,payload,refreshes,actions = transport
    credentials.access_token_expired = credentials.fail_refresh = True
    with pytest.raises(RuntimeError,match='refresh failed'): updater.upload(payload,'_state.json')
    assert len(refreshes)==1 and actions==[] and updater.receipts==[]
    assert auth.refresh_events[-1]['status']=='FAIL'


def test_permission_failure_no_retry(transport):
    credentials,auth,updater,payload,refreshes,actions = transport
    credentials.permission_error = True
    with pytest.raises(PermissionError): updater.upload(payload,'_state.json')
    assert actions==['update'] and refreshes==[] and updater.receipts==[]


def test_corrupt_readback_hard_fails(transport):
    credentials,auth,updater,payload,refreshes,actions = transport
    credentials.corrupt = True
    with pytest.raises(RuntimeError,match='Drive readback mismatch'): updater.upload(payload,'_state.json')
    assert actions==['update','readback'] and not updater.receipts[-1]['readback_pass']


@pytest.mark.parametrize('method',['LocalWebserverAuth','CommandLineAuth','GetFlow','Auth'])
def test_interactive_entrypoints_forbidden(transport, method):
    with pytest.raises(RuntimeError,match='User OAuth forbidden'): getattr(transport[1],method)()


def test_changed_principal_hard_fails(transport):
    credentials,auth,updater,payload,refreshes,actions = transport
    credentials.access_token_expired = True
    credentials.service_account_email = 'different@example.com'
    with pytest.raises(RuntimeError,match='principal changed'): updater.upload(payload,'_state.json')
    assert refreshes==[] and actions==[]


def test_scientific_primitives_ast_unchanged():
    path='scripts/run_gate_bc_seed_atomic.py'
    old=ast.parse(subprocess.check_output(['git','show','1e08060db280a38b724cc242037959fffd299261:'+path],cwd=ROOT,text=True))
    new=ast.parse((ROOT/path).read_text())
    def scientific(tree):
        return [ast.dump(n,include_attributes=False) for n in tree.body
                if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)) and n.name!='DriveUpdates']
    assert scientific(old)==scientific(new)
    for name in ('src/copeland_erdos_nets/gate_bc_protocol.py','configs/gate_bc_canonical.json','configs/gate_bc_smoke.json'):
        assert (ROOT/name).read_bytes()==subprocess.check_output(['git','show','1e08060db280a38b724cc242037959fffd299261:'+name],cwd=ROOT)
