"""Administrative attempt routing only; no scientific/RNG dependencies."""
import re

CELLS = ('B00', 'B01', 'B10', 'B11', 'C_002')
SUFFIXES = ('_state.json', '_live.csv') + tuple(
    f'_{kind}_{cell}.pt' for kind in ('checkpoint', 'best') for cell in CELLS
) + ('_archive.tar.gz', '_archive_receipt.json')
CONFIG_SHA256 = '45dea06a3afd8eb523fdb875cde0b9759a9fcfcc2c4707cbc6f816a9ad4ea8aa'


def namespace(seed, attempt_id):
    if type(seed) is not int or not 0 < seed < 2**31:
        raise ValueError('invalid transport seed')
    if not isinstance(attempt_id, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,95}', attempt_id):
        raise ValueError('invalid explicit transport attempt ID')
    return f'gate_bc_seed_{seed}_{attempt_id}'


def validate_manifest(manifest, seed, attempt_id, source_sha):
    expected = namespace(seed, attempt_id)
    if (manifest.get('version') != 1 or manifest.get('seed') != seed
        or manifest.get('attempt_id') != attempt_id or manifest.get('namespace') != expected
        or manifest.get('source_sha') != source_sha
        or not re.fullmatch(r'[0-9a-f]{40}', source_sha)
        or manifest.get('config_sha256') != CONFIG_SHA256):
        raise ValueError('transport namespace/source/config manifest mismatch')
    parent = manifest.get('exchange_id')
    if not isinstance(parent, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', parent):
        raise ValueError('invalid exchange ID')
    if not manifest.get('sa_principal') or not manifest.get('ds_authority_path') or not re.fullmatch(r'[0-9a-f]{64}', manifest.get('ds_authority_sha256', '')):
        raise ValueError('missing transport principal/DS authority')
    targets = manifest.get('targets', {})
    if set(targets) != set(SUFFIXES):
        raise ValueError('missing/extra transport target')
    ids = []
    for suffix, target in targets.items():
        fid = target.get('id')
        if (not isinstance(fid, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', fid)
            or target.get('name') != expected + suffix or target.get('parent_id') != parent):
            raise ValueError('transport target name/ID/parent mismatch')
        ids.append(fid)
    if len(set(ids)) != len(ids) or set(ids) & set(manifest.get('forbidden_ids', [])):
        raise ValueError('overlapping/duplicate attempt target ID')
    return manifest


def existing_target_id(drive, manifest, suffix):
    if suffix not in manifest['targets']:
        raise ValueError('unmapped transport suffix; no fallback')
    target = manifest['targets'][suffix]
    query = f"title='{target['name']}' and '{manifest['exchange_id']}' in parents and trashed=false"
    found = drive.ListFile({'q': query}).GetList()
    if len(found) != 1 or found[0]['id'] != target['id']:
        raise ValueError('missing/duplicate/mismatched existing transport target')
    return target['id']
