import json
from pathlib import Path
import unittest
from unittest.mock import patch

from gpu_bench.profile import public_profile, cpu_groups

try:
    import jsonschema
except ImportError:
    jsonschema = None


class ProfileTests(unittest.TestCase):
    def test_heterogeneous_cpu_models_are_preserved(self):
        rows = [{'field': 'Model name:', 'data': name} for name in ['Core A', 'Core B']]
        self.assertEqual([g['model'] for g in cpu_groups(json.dumps({'lscpu': rows}))], ['Core A', 'Core B'])

    def collect(self, names=None):
        def column(field):
            if field == "name":
                return names
            if field == "compute_cap":
                return ["8.6"] * len(names or [])
            if field == "driver_version":
                return ["580.173.02"] * len(names or [])
            return ["N/A"] * len(names or [])
        def command(argv):
            if "lscpu" in argv:
                return json.dumps({'lscpu': [{'field': 'Model name:', 'data': 'Example CPU'},
                                             {'field': 'Hostname:', 'data': 'PRIVATE_CANARY'}]}), None
            return 'PRIVATE_CANARY', None
        with patch('gpu_bench.profile.gpu_column', side_effect=column), \
             patch('gpu_bench.profile.command', side_effect=command), \
             patch('gpu_bench.profile.read_text', return_value='MemTotal: 8388608 kB'):
            return public_profile('machine-001')

    def test_identifiers_and_raw_output_not_forwarded(self):
        result = self.collect(['NVIDIA GeForce RTX 3070'])
        self.assertNotIn('PRIVATE_CANARY', json.dumps(result))
        self.assertEqual(result['system']['memory_total_gib'], 8)
        self.assertEqual(result['gpus'][0]['compute_capability'], '8.6')
        self.assertNotIn('hostname', result)
        self.assertNotIn('uuid', result['gpus'][0])

    def test_unified_memory_missing_counters(self):
        result = self.collect(['NVIDIA GB10'])
        self.assertTrue(result['capabilities']['shared_system_gpu_memory'])
        self.assertIsNone(result['gpus'][0]['memory_total_mib'])

    def test_no_gpu_is_not_claimed_as_supported(self):
        result = self.collect()
        self.assertFalse(result['capabilities']['cuda_devices_detected'])
        self.assertEqual(result['gpus'], [])

    def test_multi_gpu_retained_without_device_identifiers(self):
        result = self.collect(['NVIDIA GeForce RTX 3080', 'NVIDIA GeForce RTX 3070'])
        self.assertTrue(result['capabilities']['multi_gpu'])
        self.assertEqual([g['device'] for g in result['gpus']], ['gpu-1', 'gpu-2'])

    def test_neutral_labels_required(self):
        for label in ['my-host', '192.168.1.2', 'GPU-1234', '../machine-001']:
            with self.assertRaises(ValueError):
                public_profile(label)

    @unittest.skipUnless(jsonschema, 'Install .[test] for schema tests')
    def test_published_profiles_match_schema(self):
        schema = json.loads(Path('schemas/profile.schema.json').read_text())
        profiles = list(Path('machines').glob('machine-*.json'))
        self.assertTrue(profiles)
        for path in profiles:
            with self.subTest(path=path):
                value = json.loads(path.read_text())
                jsonschema.validate(value, schema)
                self.assertEqual(value['profile_id'], path.stem)

    @unittest.skipUnless(jsonschema, 'Install .[test] for schema tests')
    def test_schema_rejects_additional_identifiers(self):
        schema = json.loads(Path('schemas/profile.schema.json').read_text())
        result = self.collect(['NVIDIA GB10'])
        jsonschema.validate(result, schema)
        result['gpus'][0]['uuid'] = 'PRIVATE_CANARY'
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(result, schema)
