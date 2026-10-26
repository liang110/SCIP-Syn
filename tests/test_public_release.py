"""Release contracts: frozen evidence, citation version and license boundaries."""
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest
import struct
from scip_syn import __version__

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('reproduction_tools',ROOT/'tools/reproduce_manuscript.py')
reproduction=importlib.util.module_from_spec(spec)
spec.loader.exec_module(reproduction)


class PublicReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame,cls.null,cls.manifest=reproduction.load_frozen(ROOT)

    def test_complete_frozen_fold_groups(self):
        self.assertEqual(len(self.frame),self.manifest['row_count'])
        self.assertTrue(self.frame.groupby(['series','population','model','metric']).fold.nunique().eq(10).all())

    def test_source_hash_tamper_fails(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d)/'reproduce'
            folder.mkdir()
            for name in ['manifest.json',*self.manifest['files']]:
                shutil.copyfile(ROOT/'reproduce'/name,folder/name)
            with (folder/'frozen_fold_metrics.csv').open('a') as f:
                f.write('\n')
            with self.assertRaises(ValueError):
                reproduction.load_frozen(d)

    def test_control_query_effect(self):
        paired=reproduction.contrasts(self.frame[self.frame.series.eq('controls')])
        row=paired[(paired.population=='Selection-Unseen Both-Unseen') &
                   (paired.metric=='EF@5%') & (paired.control=='SCIP-Syn static query')].iloc[0]
        self.assertAlmostEqual(row.mean_delta,.509693,places=6)
        self.assertEqual((row.wins,row.ties,row.losses),(4,6,0))
        self.assertGreater(row.leave_one_fold_min,0)

    def test_negative_dynamic_film_evidence_retained(self):
        paired=reproduction.contrasts(self.frame[self.frame.series.eq('controls')])
        row=paired[(paired.population=='Selection-Unseen Both-Unseen') &
                   (paired.metric=='EF@5%') & (paired.control=='SCIP-Syn static target context')].iloc[0]
        self.assertLess(row.mean_delta,0)
        self.assertLess(row.leave_one_fold_min,0)
        self.assertGreater(row.leave_one_fold_max,0)

    def test_fixed_pool_null_is_not_fake_replication(self):
        self.assertEqual(len(self.null),10)
        self.assertTrue(self.null.fold.is_unique)
        self.assertTrue((self.null.hits<=self.null.k).all())

    def test_citation_version_and_mit(self):
        citation=(ROOT/'CITATION.cff').read_text(encoding='utf-8')
        self.assertIn(f'version: {__version__}',citation)
        self.assertIn('license: MIT',citation)
        self.assertNotIn('doi:',citation)
        self.assertTrue((ROOT/'LICENSE').read_text().startswith('MIT License'))

    def test_public_visibility_docs(self):
        for name in ['README.md','DATA.md','docs/REPRODUCTION.md']:
            text=(ROOT/name).read_text(encoding='utf-8').lower()
            self.assertNotIn('private repository',text)
            self.assertNotIn('visibility is private',text)

    def test_architecture_uses_local_png(self):
        image=(ROOT/'docs/architecture.png').read_bytes()
        self.assertTrue(image.startswith(b'\x89PNG\r\n\x1a\n'))
        width,height=struct.unpack('>II',image[16:24])
        self.assertGreater(width,0)
        self.assertGreater(height,0)
        self.assertIn('](docs/architecture.png)',(ROOT/'README.md').read_text(encoding='utf-8'))


if __name__=='__main__':
    unittest.main()

