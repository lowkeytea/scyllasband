from __future__ import annotations

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from scyllasband.cli import _parse_cli_args, _parse_group_lines, build_parser
from scyllasband.contract import BundleValidationError, validate_delivery_contract
from scyllasband.delivery import (
    DELIVERY_AXES, DELIVERY_RELEASE, DELIVERY_SCHEMA, delivery_tensors,
    resolve_delivery, validate_delivery_request,
)
from scyllasband.download import (
    download_litert_bundle, download_voice_packs, default_bundle_subdirs,
    replacement_notice, superseded_v2_bundle,
)
from scyllasband.g2p_phrases import g2p_phrase_segments
from scyllasband.planner import planner_options_from, prepare_render_chunks, _stable_delivery_key
from scyllasband.runtime import SynthesisRequest
from scyllasband.metadata_compare import compare_long_form_metadata


def measured_manifest():
    return SimpleNamespace(controls={
        'graph_input_contract': DELIVERY_SCHEMA,
        'delivery': dict(enabled=True, axes=list(DELIVERY_AXES), range=[0, 4], neutral=2,
                         whisper='binary', presence_mask=True, normalization='(rating-2)/4',
                         scope='utterance', timeline_conditioning=False),
        'span_conditioning': {'scope': 'utterance'},
    }, components={
        'duration_predictor': SimpleNamespace(inputs=('phone_ids', 'voice_id', 'language_id', 'delivery_values', 'delivery_present', 'phone_mask')),
        'vector_estimator': SimpleNamespace(inputs=('noise', 'time', 'expanded_phone_ids', 'voice_id', 'language_id', 'delivery_values', 'delivery_present', 'latent_mask', 'span_context_hidden')),
    })


class MeasuredDeliveryTest(unittest.TestCase):
    def test_neutral_is_explicit_and_auto_is_absent(self):
        neutral, present, resolved = delivery_tensors()
        automatic, absent, _ = delivery_tensors('auto')
        np.testing.assert_array_equal(neutral, automatic)
        self.assertTrue(present.all())
        self.assertFalse(absent.any())
        self.assertEqual(resolved['energy'], 2)
        self.assertEqual(resolved['whisper'], 'off')

    def test_partial_request_uses_neutral_for_unspecified_axes(self):
        values, present, resolved = delivery_tensors('energy=1.2,valence=auto,whisper=on')
        np.testing.assert_allclose(values, [[-.2, 0, 0, 0, 1]])
        np.testing.assert_array_equal(present, [[True, True, False, True, True]])
        self.assertEqual(resolved['tension'], 2)

    def test_bad_requests_are_rejected(self):
        for value in ('energy=nan', 'energy=4.01', 'energy=-.1', 'energy=1,energy=2',
                      'calm=0.5', 'whisper=.5', {'energy': True}, {'whisper': 1}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                resolve_delivery(value)

    def test_model_families_do_not_reinterpret_controls(self):
        legacy = SimpleNamespace(controls={})
        with self.assertRaises(ValueError):
            validate_delivery_request(legacy, SynthesisRequest('Hi.', 'ariadne', delivery='neutral'))
        validate_delivery_request(legacy, SynthesisRequest('Hi.', 'ariadne', affect='joy=.5'))
        for kwargs in ({'affect': 'joy=.5'}, {'emotion': 'happy'}, {'affect_guidance_scale': 2}, {'emotion_embed_scale': .5}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                validate_delivery_request(measured_manifest(), SynthesisRequest('Hi.', 'ariadne', **kwargs))

    def test_contract_cannot_claim_timeline_or_legacy_reference_routing(self):
        validate_delivery_contract(measured_manifest())
        for key, val in [('neutral', 0), ('axes', list(reversed(DELIVERY_AXES))), ('timeline_conditioning', True)]:
            m = measured_manifest(); m.controls['delivery'][key] = val
            with self.subTest(key=key), self.assertRaises(BundleValidationError):
                validate_delivery_contract(m)
        m = measured_manifest(); m.controls['reference_packs'] = {'enabled': True}
        with self.assertRaises(BundleValidationError):
            validate_delivery_contract(m)

    def test_continuity_requires_declared_training_inputs_and_passage_scope(self):
        prefix_inputs = measured_manifest().components['vector_estimator'].inputs + ('prefix_latents', 'prefix_mask')

        def continuity(trained='connected_context_v1', inputs=prefix_inputs, scope='passage'):
            m = measured_manifest()
            m.controls['prefix_conditioning'] = {'enabled': True, 'max_frames': 32, 'trained': trained}
            m.controls['span_conditioning'] = {'scope': scope}
            m.components['vector_estimator'] = SimpleNamespace(inputs=inputs)
            return m

        validate_delivery_contract(continuity())
        cases = {
            'undeclared training': continuity(trained=None),
            'missing prefix inputs': continuity(inputs=measured_manifest().components['vector_estimator'].inputs),
            'utterance scope': continuity(scope='utterance'),
        }
        for name, manifest in cases.items():
            with self.subTest(name), self.assertRaises(BundleValidationError):
                validate_delivery_contract(manifest)
        plain = measured_manifest(); plain.controls['span_conditioning'] = {'scope': 'passage'}
        with self.assertRaises(BundleValidationError):
            validate_delivery_contract(plain)
        prefixed = measured_manifest(); prefixed.components['vector_estimator'] = SimpleNamespace(inputs=prefix_inputs)
        with self.assertRaises(BundleValidationError):
            validate_delivery_contract(prefixed)

    def test_cli_and_group_planning_retain_delivery(self):
        class Runtime:
            manifest = SimpleNamespace(audio=SimpleNamespace(sample_rate=24000, latent_hop_length=512), controls={})
            def resolve_language_for_voice(self, voice, language): return language or 'en_us'
            def normalize_text(self, text, **kwargs): return text
        args = _parse_cli_args(build_parser(), ['group-speak', '--delivery', 'energy=2.8', '--text', '[ariadne:en_us:energy=1.2,whisper=on] Hello. [es] Hola.'])
        rows = _parse_group_lines(args.text_flag, default_voice=None, default_language=None)
        chunks = prepare_render_chunks(Runtime(), rows, planner_options_from(args, no_preflight_chunks=True))
        self.assertEqual(len(chunks), 2)
        for chunk in chunks:
            self.assertEqual(resolve_delivery(chunk['delivery'])['energy'], 1.2)
            self.assertEqual(resolve_delivery(chunk['delivery'])['whisper'], 'on')
            self.assertIsNone(chunk['affect'])

    def test_temporal_chain_key_accepts_binary_and_absent_coordinates(self):
        a = _stable_delivery_key({'energy': None, 'whisper': 'on'})
        self.assertEqual(a, _stable_delivery_key('energy=auto,whisper=on'))
        self.assertNotEqual(a, _stable_delivery_key({'energy': 2, 'whisper': 'on'}))

    def test_metadata_comparison_detects_control_and_presence_changes(self):
        left = {'chunks': [{'text': 'Hello.', 'metadata': {'delivery': resolve_delivery(), 'delivery_present': [True]*5}}]}
        right = {'chunks': [{'text': 'Hello.', 'metadata': {'request': {'delivery': resolve_delivery(), 'delivery_present': [True]*5}}}]}
        self.assertEqual(compare_long_form_metadata(left, right)['status'], 'ok')
        right['chunks'][0]['metadata']['request']['delivery'] = resolve_delivery('energy=auto')
        right['chunks'][0]['metadata']['request']['delivery_present'][0] = False
        self.assertEqual(compare_long_form_metadata(left, right)['status'], 'mismatch')

    def test_comma_preservation_is_opt_in(self):
        text = 'Come back, please.'
        self.assertEqual(len(g2p_phrase_segments(text)), 1)
        self.assertEqual(len(g2p_phrase_segments(text, preserve_commas=True)), 2)


class MeasuredDownloadTest(unittest.TestCase):
    def test_cached_old_v2_notices_and_v1_exclusion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for version, schema, expected in [('1', 'scyllasband_affect_v1', False), ('2', 'scyllasband_affect_v3', True), ('2', DELIVERY_SCHEMA, False)]:
                (root/'manifest.json').write_text(json.dumps({'model_version': version, 'controls': {'graph_input_contract': schema}}))
                self.assertEqual(superseded_v2_bundle(root), expected)
                self.assertEqual(replacement_notice(root) is not None, expected)

    def test_default_backend_is_measured_onnx_even_on_apple(self):
        with mock.patch('scyllasband.download.coreai_host_supported', return_value=True):
            self.assertEqual(default_bundle_subdirs(), ('onnx',))
            self.assertEqual(default_bundle_subdirs('v1'), ('coreai', 'onnx-int8'))

    def test_download_failure_preserves_old_v2_and_v1(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); old = root/'v2/onnx'; old.mkdir(parents=True)
            (old/'sentinel').write_text('old v2')
            v1 = root/'v1'; v1.mkdir(); (v1/'sentinel').write_text('v1')
            with mock.patch('scyllasband.download._require_snapshot_download', return_value=mock.Mock(side_effect=RuntimeError('offline'))):
                with self.assertRaises(RuntimeError):
                    download_litert_bundle(models_dir=root, force=True)
                with self.assertRaises(RuntimeError):
                    download_voice_packs(models_dir=root, voices_dir=old, force=True)
            self.assertEqual((old/'sentinel').read_text(), 'old v2')
            self.assertEqual((v1/'sentinel').read_text(), 'v1')

    def test_invalid_candidate_is_not_promoted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); old = root/'v2/onnx'; old.mkdir(parents=True)
            (old/'sentinel').write_text('keep')
            def fetch(**kwargs):
                candidate = Path(kwargs['local_dir'])/'onnx'; candidate.mkdir()
                (candidate/'manifest.json').write_text('{}')
                self.assertEqual(kwargs['revision'], DELIVERY_RELEASE)
            with mock.patch('scyllasband.download._require_snapshot_download', return_value=fetch):
                with self.assertRaises(BundleValidationError):
                    download_litert_bundle(models_dir=root)
            self.assertEqual((old/'sentinel').read_text(), 'keep')

    def test_success_replaces_only_selected_bundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); old = root/'v2/onnx'; old.mkdir(parents=True)
            (old/'stale.bin').write_bytes(b'stale')
            v1 = root/'v1'; v1.mkdir(); (v1/'sentinel').write_text('v1')
            def fetch(**kwargs):
                candidate = Path(kwargs['local_dir'])/'onnx'; candidate.mkdir()
                (candidate/'manifest.json').write_text('{}')
            with mock.patch('scyllasband.download._require_snapshot_download', return_value=fetch), mock.patch('scyllasband.download.validate_bundle_layout') as validate:
                self.assertEqual(download_litert_bundle(models_dir=root), old)
                validate.assert_called_once()
            self.assertFalse((old/'stale.bin').exists())
            self.assertTrue((v1/'sentinel').exists())

    def test_unavailable_measured_backend_is_explained(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch('scyllasband.download._require_snapshot_download'):
            with self.assertRaisesRegex(ValueError, 'currently ships onnx'):
                download_litert_bundle(models_dir=tmp, bundle_subdir='litert')


if __name__ == '__main__':
    unittest.main()
