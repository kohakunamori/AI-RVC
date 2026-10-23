import tempfile
import ast
import unittest
from pathlib import Path
import numpy as np
import soundfile as sf
from infer.cover_pipeline import CoverPipeline, _phase_safe_cleanup, _cached_reference


class ReferencePhaseTests(unittest.TestCase):
    def test_numpy_extrema_do_not_silently_ignore_a_third_risk(self):
        import infer.cover_pipeline as implementation
        tree=ast.parse(Path(implementation.__file__).read_text(encoding='utf-8-sig'))
        offenders=[node.lineno for node in ast.walk(tree) if isinstance(node,ast.Call)
                   and isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name)
                   and node.func.value.id=='np' and node.func.attr in ('maximum','minimum') and len(node.args)>2]
        self.assertEqual(offenders,[], 'Use an explicit out= buffer or compare all input curves')

    def signals(self):
        sr = 40000
        t = np.arange(sr * 3, dtype=np.float64) / sr
        envelope = np.interp(t, [0, .1, .9, 1.1, 1.9, 2.1, 2.9, 3], [0, 1, 1, .075, .075, 1, 1, 0])
        source = .15 * np.sin(2 * np.pi * 260 * t) * envelope
        target = .18 * np.sin(2 * np.pi * 260 * t + .6) * envelope
        target += .012 * np.sin(2 * np.pi * 5200 * t) * ((t > 1.1) & (t < 1.9))
        return sr, source.astype('float32'), target.astype('float32')

    def test_each_cleanup_is_invariant_to_reference_polarity(self):
        sr, source, target = self.signals()
        for method in ('_apply_source_breath_cleanup', '_apply_source_transition_cleanup', '_apply_artifact_segment_rescue'):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as d:
                outputs = []
                for sign in (1, -1):
                    src = Path(d) / f'source{sign}.wav'; cv = Path(d) / f'target{sign}.wav'
                    sf.write(src, source * sign, sr, subtype='FLOAT'); sf.write(cv, target, sr, subtype='FLOAT')
                    getattr(CoverPipeline(device='cpu'), method)(str(src), str(cv))
                    outputs.append(sf.read(cv, dtype='float32')[0])
                np.testing.assert_array_equal(outputs[0], outputs[1])
                self.assertGreater(np.max(np.abs(outputs[0] - target)), 1e-4, 'The cleanup must actually run')

    def test_candidate_selection_does_not_treat_reference_phase_as_timbre(self):
        sr, source, raw = self.signals()
        processed = .5 * raw + .5 * source
        with tempfile.TemporaryDirectory() as d:
            d = Path(d); rp=d/'raw.wav'; cp=d/'processed.wav'
            sf.write(rp,raw,sr,subtype='FLOAT'); sf.write(cp,processed,sr,subtype='FLOAT')
            outputs=[]
            for sign in (1,-1):
                src=d/f'source{sign}.wav';out=d/f'out{sign}.wav'
                sf.write(src,source*sign,sr,subtype='FLOAT')
                CoverPipeline(device='cpu')._apply_quality_candidate_selection(str(src),str(rp),str(cp),str(out))
                outputs.append(sf.read(out,dtype='float32')[0])
            np.testing.assert_array_equal(outputs[0],outputs[1])

    def test_source_only_frequency_is_not_injected(self):
        sr=40000;t=np.arange(sr,dtype=np.float64)/sr
        source=(.2*np.sin(2*np.pi*200*t)).astype('float32')
        target=(.2*np.sin(2*np.pi*600*t+.8)).astype('float32')
        frames=1+len(target)//512
        out=_phase_safe_cleanup(source,target,np.full(frames,.6,dtype='float32'))
        middle=out[4000:-4000];tm=t[4000:-4000]
        reference_tone=2*abs(np.mean(middle*np.exp(-2j*np.pi*200*tm)))
        self.assertLess(reference_tone,1e-5)
        self.assertLess(np.sqrt(np.mean(out*out)),np.sqrt(np.mean(target*target)))

    def test_target_polarity_and_float64_are_preserved(self):
        sr,source,target=self.signals();target=target.astype('float64')
        strength=np.full(1+len(target)//512,.4,dtype='float32')
        positive=_phase_safe_cleanup(source,target,strength)
        negative=_phase_safe_cleanup(source,-target,strength)
        self.assertEqual(positive.dtype,np.float64)
        np.testing.assert_array_equal(positive,-negative)

    def test_reference_cache_refreshes_after_file_replacement(self):
        sr,source,_=self.signals()
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'reference.wav';pipe=CoverPipeline(device='cpu')
            sf.write(p,source,44100,subtype='FLOAT')
            a,_=_cached_reference(pipe,str(p),sr)
            sf.write(p,-source,44100,subtype='FLOAT')
            b,_=_cached_reference(pipe,str(p),sr)
            np.testing.assert_array_equal(a,-b)
            self.assertFalse(a.flags.writeable)

if __name__=='__main__':unittest.main()
