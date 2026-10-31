import numpy as np
import pytest

from hushkey.preprocess import KIN_DIM, Sample, decode_wire_frame, preprocess
from hushkey.preprocess.augment import SilentSpeechAugment, attenuate
from hushkey.preprocess.keypoints import canonicalize, geometry_features, interpolate_missing, kinematic_stream
from hushkey.preprocess.mouth import mouth_rois, to_model_input
from hushkey.preprocess.pipeline import NoFaceError, RawFrame, resample_indices
from hushkey.preprocess.throat import throat_stream

from .synth import face_frame, wire_frames


def test_mirroring_cannot_swap_eyes():
    _, kp = face_frame(0.0)
    mirrored = kp.copy()
    mirrored[[0, 1]] = mirrored[[1, 0]]  # app sent eyes swapped
    mirrored[[8, 9]] = mirrored[[9, 8]]
    np.testing.assert_allclose(canonicalize(mirrored), canonicalize(kp))


def test_geometry_invariant_to_roll_and_scale():
    _, a = face_frame(0.1, iod=70, roll_deg=0)
    _, b = face_frame(0.1, iod=110, roll_deg=15, cx=170, cy=160)
    np.testing.assert_allclose(geometry_features(a), geometry_features(b), atol=1e-3)


def test_aperture_tracks_mouth_opening():
    closed = geometry_features(face_frame(0.25 / 3 * 3)[1])  # sin = -1 → minimum
    open_ = geometry_features(face_frame(0.25 / 3)[1])  # sin = +1 → maximum
    assert open_[0] > closed[0] + 0.1


def test_interpolate_missing():
    a, b = np.zeros((11, 2), np.float32), np.ones((11, 2), np.float32)
    out = interpolate_missing([None, a, None, None, b, None])
    assert np.allclose(out[0], a) and np.allclose(out[-1], b)
    assert np.allclose(out[2], 1 / 3) and np.allclose(out[3], 2 / 3)
    assert interpolate_missing([None, None]) is None


def test_mouth_roi_is_centred_on_mouth():
    frames, kps = zip(*[face_frame(i / 25) for i in range(10)])
    from hushkey.preprocess.keypoints import stable_points

    rois = mouth_rois(list(frames), [stable_points(k) for k in kps])
    assert rois.shape == (10, 96, 96) and rois.dtype == np.uint8
    dark = rois[5] < 60  # mouth interior is drawn at 30
    ys, xs = np.nonzero(dark)
    assert abs(xs.mean() - 48) < 8 and abs(ys.mean() - 48) < 12
    x = to_model_input(rois)
    assert x.shape == (10, 88, 88)


def test_throat_stream_sees_larynx_motion():
    frames, kps = zip(*[face_frame(i / 25, throat_hz=3.0) for i in range(25)])
    th = throat_stream(list(frames), list(kps))
    assert th.shape == (25, 3)
    assert th[:, 2].all()  # neck inside frame
    assert th[1:, 0].mean() > 0.002
    still = throat_stream(*zip(*[face_frame(0.0) for _ in range(5)]))
    assert still[1:, 0].mean() < th[1:, 0].mean() / 5


def test_throat_invalid_when_out_of_frame():
    frames, kps = zip(*[face_frame(i / 25, H=230) for i in range(3)])  # neck cropped off
    assert throat_stream(list(frames), list(kps))[:, 2].sum() == 0


def test_resample_30_to_25fps():
    t = [i / 30 for i in range(60)]
    idx = resample_indices(t, 25)
    assert len(idx) == 50 and idx[0] == 0 and idx[-1] == 59
    assert all(b >= a for a, b in zip(idx, idx[1:]))


def test_end_to_end_preprocess_and_roundtrip(tmp_path):
    raw = [decode_wire_frame(m) for m in wire_frames(2.0, fps=30)]
    raw[10].kp = None  # a dropped detection is interpolated
    s = preprocess(raw)
    assert s.rois.shape == (50, 96, 96)
    assert s.kin.shape == (50, KIN_DIM) and np.isfinite(s.kin).all()
    assert 0.95 < s.face_ratio < 1.0
    s.save(tmp_path / "u.npz", text="HELLO")
    s2, text = Sample.load(tmp_path / "u.npz")
    assert text == "HELLO" and np.array_equal(s2.rois, s.rois)


def test_no_face_raises():
    with pytest.raises(NoFaceError):
        preprocess([RawFrame(t=0, gray=np.zeros((10, 10), np.uint8), kp=None)])


def test_augment_shapes_and_attenuation():
    raw = [decode_wire_frame(m) for m in wire_frames(1.0, fps=25)]
    s = preprocess(raw)
    v, k = attenuate(s.rois, s.kin, alpha=0.3)
    assert np.abs(np.diff(v.astype(float), axis=0)).mean() < np.abs(np.diff(s.rois.astype(float), axis=0)).mean()
    x, kin = SilentSpeechAugment(seed=0)(s.rois, s.kin)
    assert x.shape[1:] == (88, 88) and kin.shape[0] == x.shape[0] and x.dtype == np.float32


def test_kinematic_stream_without_throat():
    kps = [face_frame(i / 25)[1] for i in range(20)]
    k = kinematic_stream(kps)
    assert k.shape == (20, KIN_DIM) and (k[:, -1] == 0).all()
