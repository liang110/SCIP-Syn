"""CPU-only contract tests. No assets downloaded and no benchmark fits run."""
import tempfile
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
import torch
from scip_syn.model import SCIPSyn, parameter_counts
from scip_syn.metrics import canonical_predictions, evaluate
from scip_syn.training import fixed_loss, scheduler
from scip_syn.data import read_frame, verify_manifest, write_json, sha256
from scip_syn.predict import load_model


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        cls.matrix = torch.randint(0, 2, (3040, 358)).float()
        cls.matrix[0] = 0
        cls.model = SCIPSyn(cls.matrix).eval()
        cls.inputs = (torch.randn(4, 768), torch.randn(4, 768),
                      torch.tensor([[1, 2, 0]] * 4), torch.tensor([[5, 9, 0]] * 4), torch.randn(4, 2000))

    def test_parameter_counts(self):
        self.assertEqual(parameter_counts(self.model), {'registered': 6651586, 'trainable': 5563266})

    def test_ab_ba_exact(self):
        a, b, ta, tb, cell = self.inputs
        with torch.no_grad():
            x = self.model(a, b, ta, tb, cell)
            y = self.model(b, a, tb, ta, cell)
        self.assertTrue(all(torch.equal(u, v) for u, v in zip(x, y)))
        self.assertEqual(x[0].shape, (4, 1))

    def test_padding_is_not_signal(self):
        a, b, ta, tb, cell = self.inputs
        with torch.no_grad():
            x = self.model(a, b, ta, tb, cell)[0]
            z = torch.zeros(4, 7, dtype=torch.long)
            y = self.model(a, b, torch.cat([ta, z], 1), torch.cat([tb, z], 1), cell)[0]
        self.assertTrue(torch.equal(x, y))

    def test_nonfinite_rejected(self):
        inputs = list(self.inputs)
        inputs[0] = inputs[0].clone()
        inputs[0][0, 0] = float('nan')
        with self.assertRaises(FloatingPointError):
            self.model(*inputs)

    def test_invalid_target_id_rejected(self):
        inputs = list(self.inputs)
        inputs[2] = torch.full((4, 3), 3040, dtype=torch.long)
        with self.assertRaises(ValueError):
            self.model(*inputs)

    def test_invalid_molecular_shape_rejected(self):
        inputs = list(self.inputs)
        inputs[0] = torch.zeros(4, 767)
        with self.assertRaises(ValueError):
            self.model(*inputs)

    def test_context_control_buffers(self):
        for variant in ['static_query', 'static_target_context']:
            model = SCIPSyn(self.matrix, variant)
            self.assertEqual(model.fusion.query_is_static.item(), int(variant == 'static_query'))
            self.assertEqual(model.target_context_is_static.item(), int(variant == 'static_target_context'))

    def test_target_off_buffer(self):
        self.assertEqual(SCIPSyn(self.matrix, 'target_input_off').target_information_enabled.item(), 0)

    def test_frozen_mapping(self):
        self.assertFalse(self.model.target_tower.base_tower.pathway_projection.weight.requires_grad)
        self.assertTrue(torch.equal(self.model.target_tower.base_tower.pathway_projection.weight, self.matrix.t()))

    def test_nonfinite_hidden_output_rejected(self):
        model = SCIPSyn(self.matrix).eval()
        with torch.no_grad():
            model.molecular_projection[0].weight[0, 0] = float('inf')
        with self.assertRaises(FloatingPointError):
            model(*self.inputs)

    def test_checkpoint_hash_required_before_loading(self):
        with self.assertRaises(ValueError):
            load_model('not-read-without-trusted-hash.pt', None, 'cpu')

    def test_checkpoint_round_trip(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'owned_checkpoint.pt'
            torch.save({'variant': 'full', 'model_state': self.model.state_dict()}, path)
            model, _ = load_model(path, SimpleNamespace(matrix=self.matrix), 'cpu',
                                  expected_checkpoint_sha256=sha256(path))
            model.eval()
            with torch.no_grad():
                self.assertTrue(torch.equal(self.model(*self.inputs)[0], model(*self.inputs)[0]))


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({'drug1': [f'A{i:03}' for i in range(100)], 'drug2': ['Z'] * 100,
            'cell_line_id': ['C'] * 100, 'y_true_reg': [20.0] * 10 + [0.0] * 90,
            'y_pred_reg': np.arange(100, 0, -1, dtype=float)})

    def test_known_retrieval(self):
        result = evaluate(self.frame)
        self.assertEqual(result['top_5pct']['hits'], 5)
        self.assertEqual(result['top_5pct']['EF'], 10)
        self.assertEqual(result['top_1pct']['budget'], 1)

    def test_canonical_mean(self):
        x = self.frame.iloc[:1].copy()
        y = x.copy()
        y[['drug1', 'drug2']] = y[['drug2', 'drug1']].to_numpy()
        y.y_pred_reg = 50
        out = canonical_predictions(pd.concat([x, y]))
        self.assertEqual(len(out), 1)
        self.assertEqual(out.y_pred_reg.iloc[0], 75)

    def test_complete_outer_train_exposure(self):
        result = evaluate(self.frame, {'A000', 'A001'}, 'both_unseen')
        self.assertEqual(result['N'], 98)
        self.assertEqual(evaluate(self.frame, {'Z'}, 'one_unseen')['N'], 100)

    def test_missing_exposure_reference_rejected(self):
        with self.assertRaises(ValueError):
            evaluate(self.frame, population='both_unseen')

    def test_constant_prediction_reports_null(self):
        self.frame.y_pred_reg = 1
        self.assertIsNone(evaluate(self.frame)['PCC'])

    def test_zero_positive_ef_is_undefined(self):
        self.frame.y_true_reg = 0
        self.assertIsNone(evaluate(self.frame)['top_5pct']['EF'])

    def test_nonfinite_metrics_rejected(self):
        self.frame.loc[0, 'y_pred_reg'] = np.inf
        with self.assertRaises(ValueError):
            evaluate(self.frame)

    def test_manifest_fail_closed(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'input.txt'
            p.write_text('frozen', encoding='utf-8')
            manifest = Path(d) / 'hashes.json'
            write_json(manifest, {'fit': sha256(p)})
            verify_manifest(manifest, {'fit': p})
            p.write_text('drift', encoding='utf-8')
            with self.assertRaises(ValueError):
                verify_manifest(manifest, {'fit': p})

    def test_loss_and_scheduler(self):
        out = fixed_loss(torch.tensor([[1.], [2.]]), torch.tensor([0., 2.]),
                         torch.zeros(2, 1), torch.tensor([0., 1.]))
        self.assertAlmostEqual(float(out), 2.5 + np.log(2), places=6)
        p = torch.nn.Parameter(torch.ones(1))
        opt = torch.optim.Adam([p], lr=1)
        sched = scheduler(opt, 2, 10)
        self.assertEqual(opt.param_groups[0]['lr'], 0)
        opt.step(); sched.step()
        self.assertEqual(opt.param_groups[0]['lr'], .5)


if __name__ == '__main__':
    unittest.main()
