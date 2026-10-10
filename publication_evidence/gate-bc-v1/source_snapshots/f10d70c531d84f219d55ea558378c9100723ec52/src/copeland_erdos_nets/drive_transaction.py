"""Bounded same-ID, frozen-byte Drive update/readback transactions."""
import hashlib
import shutil
import tempfile
import time
from pathlib import Path

from googleapiclient.errors import HttpError
from pydrive2.files import ApiRequestError


def http_status(error):
    if isinstance(error, ApiRequestError):
        error = error.args[0] if error.args else None
    if isinstance(error, HttpError):
        return int(error.resp.status)
    return None


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def update_readback(drive, target_id, path, workdir, record, sleeper=time.sleep):
    """One shared three-retry budget; callers cannot progress before return.

    PyDrive2 Upload uses googleapiclient execute(num_retries=0 default).
    GetContentFile uses next_chunk() with num_retries=0 default per chunk.
    Fresh file objects prevent reuse of a failed resumable request state.
    The frozen local copy never changes, including after ambiguous commits.
    """
    receipt = {'target_id': target_id, 'artifact': Path(path).name,
               'retries': [], 'outcome': 'PENDING', 'readback_pass': False}
    with tempfile.TemporaryDirectory(prefix='.drive_transaction_', dir=workdir) as temp:
        frozen, readback = Path(temp)/'payload', Path(temp)/'readback'
        shutil.copyfile(path, frozen)
        receipt.update(payload_sha256=digest(frozen), size=frozen.stat().st_size)
        used = 0

        def operation(phase, call):
            nonlocal used
            while True:
                try:
                    return call()
                except Exception as error:
                    status = http_status(error)
                    if status not in (500, 502, 503, 504) or used == 3:
                        receipt.update(outcome='HARD_FAILURE', failed_phase=phase,
                                       terminal_http_status=status)
                        raise
                    delay = (1, 2, 4)[used]
                    used += 1
                    receipt['retries'].append({'phase': phase, 'retry_index': used,
                                               'http_status': status, 'delay_seconds': delay})
                    sleeper(delay)

        def upload():
            target = drive.CreateFile({'id': target_id})
            target.SetContentFile(str(frozen))
            target.Upload()

        def download():
            drive.CreateFile({'id': target_id}).GetContentFile(str(readback))

        try:
            operation('update', upload)
            operation('readback', download)
            receipt['readback_sha256'] = digest(readback)
            receipt['readback_pass'] = (receipt['readback_sha256'] == receipt['payload_sha256']
                                        and readback.stat().st_size == receipt['size'])
            if not receipt['readback_pass']:
                receipt.update(outcome='HARD_FAILURE', failed_phase='verify')
                raise RuntimeError('Drive readback mismatch (byte/hash) STOP_FOR_DS')
            receipt['outcome'] = 'PASS'
            return receipt
        finally:
            if receipt['outcome'] == 'PENDING':
                receipt['outcome'] = 'HARD_FAILURE'
            record(receipt)
