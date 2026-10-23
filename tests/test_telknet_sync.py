"""Behavioral regressions for the TelkNet October audio updates."""
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import soundfile as sf


class TelkNetSyncTests(unittest.TestCase):
    def test_constant_pitch_has_no_frame_rate_sidebands(self):
        import torch
        from infer.lib.infer_pack.models import SineGen
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            for frequency in (220.,440.,880.):
                with self.subTest(dtype=dtype,frequency=frequency):
                    pitch=torch.full((1,200,1),frequency,dtype=dtype)
                    wave=SineGen(40000)._f02sine(pitch,400).reshape(-1)
                    self.assertEqual(wave.dtype,torch.float32)
                    spectrum=torch.fft.rfft(wave.double()*torch.hann_window(len(wave),periodic=False,dtype=torch.float64)).abs().square()
                    frequencies=torch.fft.rfftfreq(len(wave),d=1/40000)
                    tone=(frequencies-frequency).abs() <= 10
                    self.assertLess(10*torch.log10(spectrum[~tone].sum()/spectrum[tone].sum()),-95)

    def test_oscillator_keeps_pitch_and_excitation_random_stream(self):
        import torch
        from infer.lib.infer_pack.models import SineGen
        for harmonics in (0,3):
            for dtype in (torch.float32,torch.float64):
                with self.subTest(harmonics=harmonics,dtype=dtype):
                    pitch=torch.linspace(130.,700.,200,dtype=dtype).reshape(1,-1,1)
                    original=pitch.clone()
                    torch.manual_seed(611011)
                    torch.rand(1,1,harmonics+1,dtype=dtype)
                    expected=torch.rand(10)
                    torch.manual_seed(611011)
                    wave=SineGen(40000,harmonic_num=harmonics)._f02sine(pitch,400)
                    torch.testing.assert_close(torch.rand(10),expected,rtol=0,atol=0)
                    torch.testing.assert_close(pitch,original,rtol=0,atol=0)
                    self.assertEqual(wave.shape,(1,80000,harmonics+1))
                    self.assertTrue(torch.isfinite(wave).all())

    def test_vocals_mode_decodes_float64_and_bypasses_every_song_stage(self):
        from infer import cover_pipeline as module
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'input.wav'
            signal = np.column_stack((np.sin(np.arange(8193) * .02), np.cos(np.arange(8193) * .02))) * .1
            sf.write(source, signal, 44100, subtype='DOUBLE')
            pipeline = module.CoverPipeline('cpu')
            pipeline.temp_dir = root / 'sessions'
            def convert(**args):
                audio, sr = sf.read(args['vocals_path'], dtype='float64', always_2d=True)
                np.testing.assert_array_equal(audio, signal)
                sf.write(args['output_path'], audio * .5, sr, subtype='DOUBLE')
            with (
                patch('infer.rvc_checkpoint.safe_load_rvc_checkpoint', return_value={}),
                patch('infer.contracts.inspect_checkpoint', return_value=SimpleNamespace(speaker_count=2, uses_f0=True)),
                patch.object(module, 'convert_vocals_official_upstream', side_effect=convert) as vc,
                patch.object(pipeline, '_init_separator', side_effect=AssertionError('separation called')),
                patch.object(pipeline, '_prepare_vocals_for_vc', side_effect=AssertionError('dereverb called')),
                patch.object(pipeline, '_separate_karaoke', side_effect=AssertionError('karaoke called')),
                patch.object(pipeline, '_constrain_converted_to_source', side_effect=AssertionError('cleanup called')),
                patch.object(module, 'mix_vocals_and_accompaniment', side_effect=AssertionError('mix called')),
            ):
                result = pipeline.process(str(source), 'test.pth', index_ratio=0,
                    input_mode='vocals', pitch_shift=3, speaker_id=1, protect=.2,
                    output_dir=str(root / 'results'))
            self.assertEqual(set(result), {'vocals', 'converted_vocals', 'all_files_dir'})
            self.assertEqual(vc.call_args.kwargs['speaker_id'], 1)
            self.assertEqual(vc.call_args.kwargs['pitch_shift'], 3)
            self.assertEqual(vc.call_args.kwargs['protect'], .2)
            for key in ('vocals', 'converted_vocals'):
                info = sf.info(result[key])
                self.assertEqual((info.frames, info.samplerate, info.channels, info.subtype), (8193,44100,2,'DOUBLE'))

    def test_invalid_input_mode_fails_before_creating_a_session(self):
        from infer.cover_pipeline import CoverPipeline
        pipeline = CoverPipeline('cpu')
        with patch.object(pipeline, '_get_session_dir', side_effect=AssertionError('session created')):
            with self.assertRaisesRegex(ValueError, 'input_mode'):
                pipeline.process('missing.wav', 'missing.pth', input_mode='unknown')

    def test_chunking_preserves_absolute_samples_despite_shortened_render_padding(self):
        from lib.rvc_chunking import render_aligned_segments
        source = np.sin(np.arange(128017, dtype=np.float64) * .073) * .1
        def render(model, net, sid, audio, *args):
            return np.repeat(audio, 2)[:-160]
        pipeline = SimpleNamespace(window=160, t_pad2=32000, t_pad_tgt=32000, vc=render)
        padded = np.pad(source, (16000,16000), mode='reflect')
        for cuts in ([], [32000], [32000,64000,96000], [160,32000,64000,128000]):
            with self.subTest(cuts=cuts):
                actual = render_aligned_segments(pipeline, None,None,0,source,padded,None,None,
                    [0,0,0],None,None,0,'v2',.33,32000,cuts)
                self.assertEqual(actual.dtype, np.float64)
                np.testing.assert_allclose(actual, np.repeat(source,2), rtol=0, atol=2e-17)

    def test_local_chunk_join_does_not_shift_or_amplify_identical_overlap(self):
        from infer.pipeline import VoiceConversionPipeline
        pipeline = VoiceConversionPipeline('cpu')
        pipeline.output_sr = 40000
        source = np.sin(np.arange(120000, dtype=np.float64) * .017) * .2
        chunks = [source[:60000], source[44000:104000], source[88000:]]
        actual = pipeline._crossfade_chunks(chunks, 20)
        np.testing.assert_allclose(actual, source, rtol=0, atol=3e-17)

    def test_karaoke_rejects_mismatched_specs_and_nonfinite_samples(self):
        from infer.separator import _validate_karaoke_audio
        with tempfile.TemporaryDirectory() as directory:
            source, candidate = Path(directory)/'source.wav', Path(directory)/'candidate.wav'
            sf.write(source, np.ones((1024,2)),44100,subtype='FLOAT')
            reference=sf.info(source)
            for audio, sr in ((np.ones((1000,2)),44100), (np.ones((1024,1)),44100),
                              (np.ones((1024,2)),48000), (np.full((1024,2), np.nan),44100)):
                with self.subTest(shape=audio.shape, sr=sr):
                    sf.write(candidate,audio,sr,subtype='FLOAT')
                    with self.assertRaises(RuntimeError):
                        _validate_karaoke_audio(str(candidate),reference)

    def test_browser_loopback_bypasses_proxy_and_preserves_existing_exclusions(self):
        from ui.server_runtime import configure_browser_server
        with patch.dict(os.environ, {'NO_PROXY':'example.test','no_proxy':'internal.test'}):
            expected={'localhost','127.0.0.1','::1'}
            for name in ('NO_PROXY', 'no_proxy'):
                expected.update(os.environ[name].split(','))
            configure_browser_server()
            self.assertEqual(set(os.environ['NO_PROXY'].split(',')),expected)
            self.assertEqual(os.environ['NO_PROXY'],os.environ['no_proxy'])

    def test_float64_reverb_matches_juce_without_narrowing(self):
        from lib.mixer import apply_reverb
        audio=np.zeros((2,48000),dtype=np.float64)
        audio[:,0]=[.200000000000001,.100000000000001]
        double=apply_reverb(audio,48000,room_size=.3,wet_level=.2)
        single=apply_reverb(audio.astype(np.float32),48000,room_size=.3,wet_level=.2)
        self.assertEqual(double.dtype,np.float64)
        self.assertEqual(double.shape,audio.shape)
        self.assertTrue(np.isfinite(double).all())
        self.assertTrue(np.any(double[:,1000:]))
        np.testing.assert_allclose(double,single,rtol=2e-4,atol=1e-7)


if __name__ == '__main__':
    unittest.main()
