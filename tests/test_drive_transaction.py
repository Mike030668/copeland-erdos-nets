import json
from pathlib import Path

import httplib2
import pytest
from googleapiclient.errors import HttpError
from pydrive2.files import ApiRequestError

from copeland_erdos_nets.drive_transaction import update_readback


def fault(status, wrapped=False):
    error = HttpError(httplib2.Response({'status': str(status)}),
                      json.dumps({'error': {'code': status}}).encode())
    return ApiRequestError(error) if wrapped else error


class Drive:
    def __init__(self, updates=(), reads=(), ambiguous=False, corrupt=False):
        self.updates, self.reads = list(updates), list(reads)
        self.ambiguous, self.corrupt = ambiguous, corrupt
        self.writes, self.ids, self.remote = [], [], b''

    def CreateFile(self, metadata):
        assert metadata == {'id': 'existing-id'}
        self.ids.append(metadata['id'])
        drive = self

        class File:
            def SetContentFile(self, path):
                self.payload = Path(path).read_bytes()

            def Upload(self):
                drive.writes.append(self.payload)
                error = drive.updates.pop(0) if drive.updates else None
                if error is None or drive.ambiguous:
                    drive.remote = self.payload
                if error:
                    raise error

            def GetContentFile(self, path):
                error = drive.reads.pop(0) if drive.reads else None
                if error:
                    raise error
                Path(path).write_bytes(b'corrupt' if drive.corrupt else drive.remote)
        return File()


def execute(tmp_path, drive):
    source = tmp_path/'source'
    source.write_bytes(b'frozen payload')
    receipts, delays = [], []
    result = update_readback(drive, 'existing-id', source, tmp_path,
                             receipts.append, delays.append)
    assert receipts == [result]
    return result, delays


def test_normal(tmp_path):
    result, delays = execute(tmp_path, Drive())
    assert result['outcome'] == 'PASS' and delays == []


@pytest.mark.parametrize('status', [500, 502, 503, 504])
@pytest.mark.parametrize('phase', ['update', 'readback'])
@pytest.mark.parametrize('wrapped', [False, True])
def test_allowlist(tmp_path, status, phase, wrapped):
    drive = Drive(updates=[fault(status, wrapped)] if phase == 'update' else [],
                  reads=[fault(status, wrapped)] if phase == 'readback' else [])
    result, delays = execute(tmp_path, drive)
    assert delays == [1] and result['retries'][0]['phase'] == phase
    assert len(drive.writes) == (2 if phase == 'update' else 1)
    assert set(drive.writes) == {b'frozen payload'}


def test_ambiguous_commit_and_shared_budget(tmp_path):
    drive = Drive(updates=[fault(500)], reads=[fault(502), fault(503)], ambiguous=True)
    result, delays = execute(tmp_path, drive)
    assert delays == [1, 2, 4] and result['outcome'] == 'PASS'
    assert drive.writes == [b'frozen payload'] * 2


def test_exhaustion_shared(tmp_path):
    source = tmp_path/'source'; source.write_bytes(b'frozen')
    receipts, delays = [], []
    drive = Drive(updates=[fault(500)], reads=[fault(502)] * 3)
    with pytest.raises(HttpError):
        update_readback(drive, 'existing-id', source, tmp_path, receipts.append, delays.append)
    assert delays == [1, 2, 4]
    assert receipts[0]['outcome'] == 'HARD_FAILURE'
    assert len(drive.writes) == 2


@pytest.mark.parametrize('error', [fault(400), fault(401), fault(403), fault(404),
                                  fault(429), fault(501), TimeoutError('500'),
                                  RuntimeError('HTTP 503')])
def test_nonretryable(tmp_path, error):
    source = tmp_path/'source'; source.write_bytes(b'frozen')
    receipts, delays = [], []
    with pytest.raises(type(error)):
        update_readback(Drive(updates=[error]), 'existing-id', source,
                        tmp_path, receipts.append, delays.append)
    assert delays == [] and receipts[0]['outcome'] == 'HARD_FAILURE'


def test_corruption_no_retry(tmp_path):
    with pytest.raises(RuntimeError, match='byte/hash'):
        execute(tmp_path, Drive(corrupt=True))


def test_no_progress_and_frozen_source(tmp_path):
    source = tmp_path/'source'; source.write_bytes(b'frozen')
    drive = Drive(updates=[fault(500)], ambiguous=True)
    progress = []
    def wait(delay):
        assert progress == []
        source.write_bytes(b'changed externally')
    receipt = update_readback(drive, 'existing-id', source, tmp_path, lambda r: None, wait)
    progress.append('after verified transaction')
    assert receipt['outcome'] == 'PASS' and drive.writes == [b'frozen'] * 2
