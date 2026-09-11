"""Regression checks for scientific table corruption and metric definitions."""

import csv
import io
import unittest

import numpy as np

from experiments.rebuild_research_tables import csv_text
from experiments.audit_perceptual_evidence import energy_metrics, paired_summary


class EvidenceTests(unittest.TestCase):
    def test_csv_commas_quotes_newlines_roundtrip(self):
        row = {'crop': 'RandomCrop(0.3, 1.0)', 'loss': 'mean MSE, selected by score',
               'description': '"原图"\n续训', 'alpha': 2}
        result = list(csv.DictReader(io.StringIO(csv_text([row]))))
        self.assertEqual(result, [{k: str(v) for k, v in row.items()}])
        self.assertNotIn(None, result[0])

    def test_concentration_uniform_and_single_spike(self):
        uniform = energy_metrics(np.ones((1,16)))
        self.assertEqual(uniform['gini'][0], 0)
        self.assertEqual(uniform['cv'][0], 0)
        self.assertEqual(uniform['top10_energy_share'][0], 2/16)
        spike = np.zeros((1,16))
        spike[0,0] = 1
        actual = energy_metrics(spike)
        self.assertEqual(actual['gini'][0], 15/16)
        self.assertEqual(actual['top10_energy_share'][0], 1)
        for key, val in actual.items():
            np.testing.assert_allclose(val, energy_metrics(spike*20)[key])

    def test_paired_change_sign_and_fraction(self):
        actual = paired_summary([1,2,3,4], [0,3,2,4])
        self.assertEqual(actual['percent_lower'], 50)
        self.assertEqual(actual['median_change'], -0.5)


if __name__ == '__main__':
    unittest.main()
