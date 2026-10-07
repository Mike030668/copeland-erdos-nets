"""CPU synthetic enablement checks. Reserved numbers are gate/metadata only."""
import copy
import itertools
import json
from pathlib import Path
import random
import types
import importlib.util

import numpy as np
import pytest
import torch

from copeland_erdos_nets.gate_bc_protocol import execution_gate, HardGateError
from copeland_erdos_nets.r010_protocol import derive_seeds

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('gate_bc_test_fixtures', ROOT/'tests/test_gate_bc_protocol.py')
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)
fixture, module = helpers.fixture, helpers.module


def test_execution_matrix_before_rng():
    cpu, numpy, python = torch.get_rng_state().clone(), np.random.get_state(), random.getstate()
    accepted = []
    for seed, mode, epochs in itertools.product([7, 66, 67, 68, 69, 70, 71, 72, 1067], ['smoke', 'canonical', 'bad'], [0, 1, 14, 15, 16]):
        allowed = (seed == 1067 and mode == 'smoke' and epochs == 1) or (seed in range(67,72) and mode == 'canonical' and epochs == 15)
        if allowed:
            execution_gate(seed, mode, epochs)
            accepted.append((seed, mode, epochs))
        else:
            with pytest.raises(HardGateError):
                execution_gate(seed, mode, epochs)
    assert len(accepted) == 6
    assert torch.equal(cpu, torch.get_rng_state())
    assert np.array_equal(numpy[1], np.random.get_state()[1]) and python == random.getstate()


def test_config_only_authorized_differences():
    runner = module(ROOT/'scripts/run_gate_bc_seed_atomic.py')
    smoke = json.loads((ROOT/'configs/gate_bc_smoke.json').read_text())
    canonical = json.loads((ROOT/'configs/gate_bc_canonical.json').read_text())
    runner.config_gate(smoke, 'smoke', 1067)
    runner.config_gate(canonical, 'canonical', 67)
    for key in ['data', 'model', 'runtime_freeze', 'rng_policy', 'parity', 'telemetry']:
        assert canonical[key] == smoke[key]
    training = dict(canonical['training'], epochs=1, held_out_test=False)
    assert training == smoke['training']
    with pytest.raises(HardGateError): runner.config_gate(canonical, 'smoke', 1067)
    with pytest.raises(HardGateError): runner.config_gate(smoke, 'canonical', 67)
    for section, key, value in [('training','lr',1), ('training','held_out_test',False), ('experiment','seeds',[67])]:
        bad = copy.deepcopy(canonical); bad[section][key] = value
        with pytest.raises(HardGateError): runner.config_gate(bad, 'canonical', 67)


def test_invalid_main_stops_before_data_model_rng(tmp_path, monkeypatch):
    runner = module(ROOT/'scripts/run_gate_bc_seed_atomic.py')
    for name in ['runtime_gate', 'load_data', 'derive_seeds', 'build_base_state']:
        monkeypatch.setattr(runner, name, lambda *a, **k: pytest.fail('invalid gate reached activity'))
    monkeypatch.setattr('sys.argv', ['runner', '--config', str(ROOT/'configs/gate_bc_canonical.json'), '--output', str(tmp_path), '--seed', '72', '--mode', 'canonical'])
    before = torch.get_rng_state().clone()
    with pytest.raises(HardGateError): runner.main()
    assert torch.equal(before, torch.get_rng_state()) and not (tmp_path/'attempt.json').exists()


def test_synthetic_canonical_main_five_cells_15epochs(tmp_path, monkeypatch):
    runner = module(ROOT/'scripts/run_gate_bc_seed_atomic.py')
    cfg = json.loads((ROOT/'configs/gate_bc_canonical.json').read_text())
    equivalence = runner.telemetry_equivalence_receipt()
    monkeypatch.setattr(runner, 'telemetry_equivalence_receipt', lambda: equivalence)
    # Config/authorization stays production-exact. Only dependencies/data/model
    # are synthetic: seed7, tiny CPU model, no provider/HF/Drive access.
    ids = torch.arange(48).reshape(12,4) % 17
    dataset = torch.utils.data.TensorDataset(ids, (ids+1)%17)
    calls = {'train': 0, 'validation': 0, 'test': 0}
    class HeldOut(torch.utils.data.TensorDataset):
        def __len__(self):
            assert calls['train'] % 15 == 0 and calls['train'] > 0
            return super().__len__()
        def __getitem__(self, index):
            assert calls['test'] > 0 and calls['train'] % 15 == 0
            return super().__getitem__(index)
    heldout = HeldOut(ids, (ids+1)%17)
    monkeypatch.setattr(runner, 'runtime_gate', lambda *a: None)
    monkeypatch.setattr(runner, 'load_data', lambda *a: ({'train':dataset,'validation':dataset,'test':heldout},17))
    monkeypatch.setattr(runner, 'derive_seeds', lambda numeric_seed: derive_seeds(7))
    def synthetic_base(*args, **kwargs):
        model_class = type(fixture()[0])
        with torch.random.fork_rng():
            torch.manual_seed(7)
            base = model_class(vocab_size=17, d_model=8, n_heads=2, d_ff=16, n_layers=1, max_seq_len=128)
        return base, None
    monkeypatch.setattr(runner, 'build_base_state', synthetic_base)
    # Keep production batchsize32: create exactly64 tiny samples below.
    ids = torch.arange(256).reshape(64,4)%17
    dataset = torch.utils.data.TensorDataset(ids,(ids+1)%17)
    heldout = HeldOut(ids,(ids+1)%17)
    real_torch = runner.torch
    class CPUOnly:
        def __getattr__(self, key): return getattr(real_torch,key)
        def device(self, *args, **kwargs): return real_torch.device('cpu')
    monkeypatch.setattr(runner, 'torch', CPUOnly())
    real_train, real_validation, real_endpoint = runner.train_epoch, runner.validation_loss, runner.held_out_endpoint
    def training(*a, **k):
        calls['train'] += 1
        return real_train(*a, **k)
    def validation(model, loader, device):
        if loader.dataset is heldout:
            return real_validation(model,loader,device)
        calls['validation'] += 1
        # Validation plateau: earliest strict improvement is epoch2, not15.
        return 4.0 if calls['validation']%15 == 1 else 3.0
    def endpoint(model, loader, device):
        assert calls['train'] == (calls['test']+1)*15
        calls['test'] += 1
        names = ['B00','B01','B10','B11','C_002']
        best = torch.load(tmp_path/'checkpoints'/f'{names[calls["test"]-1]}_best.pt', weights_only=True)
        assert best['epoch'] == 2
        assert all(torch.equal(v,model.state_dict()[k]) for k,v in best['model'].items())
        return real_endpoint(model,loader,device)
    monkeypatch.setattr(runner,'train_epoch',training)
    monkeypatch.setattr(runner,'validation_loss',validation)
    monkeypatch.setattr(runner,'held_out_endpoint',endpoint)
    monkeypatch.setattr('sys.argv', ['runner','--config',str(ROOT/'configs/gate_bc_canonical.json'),'--output',str(tmp_path),'--seed','67','--mode','canonical','--attempt-id','SYNTHETIC_CPU_NOT_SEED67'])
    monkeypatch.setenv('GATE_BC_SOURCE_SHA','synthetic-integration-source')
    monkeypatch.delenv('GATE_BC_DRIVE_LIVE',raising=False)
    runner.main()
    assert calls == {'train':75,'validation':75,'test':5}
    assert json.loads((tmp_path/'completion.json').read_text())['status'] == 'CANONICAL_COMPLETE'
    assert json.loads((tmp_path/'state.json').read_text())['total_epochs'] == 15
    gates = json.loads((tmp_path/'parity_gates.json').read_text()); assert all(x['pass'] for x in gates)
    import csv
    live = list(csv.DictReader((tmp_path/'live.csv').open()))
    endpoints = list(csv.DictReader((tmp_path/'metrics.csv').open()))
    assert len(live) == 75 and len(endpoints) == 5
    pre_gate = list(csv.DictReader((tmp_path/'batch_order_pre_gate.csv').open()))
    assert len(pre_gate) == 75
    for row in live:
        expected = next(r for r in pre_gate if r['condition']==row['condition'] and r['epoch']==row['epoch'])
        assert row['actual_batch_order_sha256'] == expected['batch_order_sha256']
    assert all(r['status']=='CANONICAL' for r in live + endpoints)
    assert all(r['test_evaluations']=='1' and r['best_epoch']=='2' for r in endpoints)
    for name in ['B00','B01','B10','B11','C_002']:
        current = torch.load(tmp_path/'checkpoints'/f'{name}_current.pt',weights_only=False)
        best = torch.load(tmp_path/'checkpoints'/f'{name}_best.pt',weights_only=True)
        assert current['epoch']==15 and best['epoch']==2
        assert current['status']==best['status']=='CANONICAL'
        assert current['attempt_id']==best['attempt_id']=='SYNTHETIC_CPU_NOT_SEED67'
        assert current['source_sha']==best['source_sha']=='synthetic-integration-source'
        assert current['rng'] and current['optimizer']['state']
    receipt = {'status':'PASS','synthetic_cpu_only':True,'rng_fixture_seed':7,'reserved_seed_argument':'metadata/gate only; derive_seeds replaced with7',
               'train_epochs':calls['train'],'epochs_per_condition':15,'conditions':5,'held_out_calls':5,
               'selected_epoch_all_cells':2,'telemetry_rows':75,'parity_gates':len(gates),'canonical_seeds_consumed':False}
    runner.atomic_json(ROOT/'.tmp/gate_bc_static/precanonical_integration.json',receipt)


def test_scientific_primitives_unchanged_from_reviewed_smoke():
    import ast
    import subprocess
    baseline = '4d7b3d937e143170c290f123d0654d3ca1c76b35'
    expected = {
        'src/copeland_erdos_nets/gate_bc_protocol.py': ['TokenScaleMixin','construct_cells','audit_cells','batch_schedule_receipt','gradient_stats','epoch_telemetry','atomic_json','atomic_checkpoint'],
        'scripts/run_gate_bc_seed_atomic.py': ['runtime_gate','load_data','validation_loss','train_epoch','rng_state','telemetry_equivalence_receipt'],
    }
    for path, names in expected.items():
        old = ast.parse(subprocess.check_output(['git','show',baseline+':'+path],cwd=ROOT,text=True))
        new = ast.parse((ROOT/path).read_text())
        for name in names:
            a = next(n for n in old.body if getattr(n,'name',None)==name)
            b = next(n for n in new.body if getattr(n,'name',None)==name)
            assert ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False), name
