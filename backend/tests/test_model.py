import json

import numpy as np
import torch

from hushkey.data import UtteranceDataset, collate
from hushkey.model import Decoder, HushNet, ctc_greedy
from hushkey.model.lora import LoRALinear, set_lora_enabled
from hushkey.model.mwer import mwer_loss, word_edit_distance
from hushkey.model.tokenizer import Tokenizer, normalize_text
from hushkey.preprocess import decode_wire_frame, preprocess

from .synth import wire_frames


def tiny(**kw):
    torch.manual_seed(0)
    return HushNet.from_pretrained(tiny=True, **kw).eval()


def test_untrained_adapters_are_exact_identity():
    """Zero-init LoRA-B and a zero gate mean HushNet == the pretrained backbone at step 0."""
    net = tiny()
    x, k = torch.randn(1, 20, 88, 88), torch.randn(1, 20, 16)
    with torch.no_grad():
        a, _ = net.encode(x, k)
        set_lora_enabled(net, False)
        b, _ = net.encode(x, None)
    assert torch.allclose(a, b, atol=1e-5)


def test_lora_targets_and_trainable_fraction():
    net = tiny()
    names = net.lora_targets
    assert any("self_attn.linear_q" in n for n in names)
    assert any("feed_forward_macaron.w_1" in n for n in names)
    assert any(n.startswith("decoder.") for n in names)
    assert all(isinstance(net.backbone.get_submodule(n), LoRALinear) for n in names)
    trainable = {n for n, p in net.named_parameters() if p.requires_grad}
    assert trainable and all(".lora_" in n or n.startswith("fusion.") for n in trainable)


def test_adapter_roundtrip(tmp_path):
    net = tiny()
    with torch.no_grad():
        for p in net.trainable_parameters():
            p.add_(torch.randn_like(p) * 0.05)
    net.save_adapters(tmp_path / "a.pt")
    net2 = tiny(adapter_path=tmp_path / "a.pt")
    x, k = torch.randn(1, 15, 88, 88), torch.randn(1, 15, 16)
    with torch.no_grad():
        assert torch.allclose(net.encode(x, k)[0], net2.encode(x, k)[0], atol=1e-5)


def test_loss_backward_and_decode():
    net = tiny()
    net.train()
    ys = torch.full((2, 4), -1)
    ys[0, :4] = torch.tensor([5, 6, 7, 8])
    ys[1, :2] = torch.tensor([9, 10])
    out = net.loss(torch.randn(2, 25, 88, 88), torch.randn(2, 25, 16), torch.tensor([25, 18]), ys)
    out["loss"].backward()
    assert net.fusion.gate.grad is not None and net.fusion.gate.grad.abs().sum() > 0
    net.eval()
    hs, _ = net.encode(torch.randn(1, 25, 88, 88), torch.randn(1, 25, 16))
    hyps = Decoder(net, beam_size=3).beam(hs[0], nbest=3)
    assert hyps and isinstance(hyps[0].text, str)
    assert isinstance(ctc_greedy(net, hs[0]), str)


def test_mwer_gradient_prefers_lower_error():
    net = tiny()
    hs, _ = net.encode(torch.randn(1, 25, 88, 88), torch.randn(1, 25, 16))
    tok = Tokenizer()
    good, bad = tok.encode("HELLO WORLD").tolist(), tok.encode("YELLOW WORD").tolist()
    loss, st = mwer_loss(net, hs[0], [good, bad], ["HELLO WORLD", "YELLOW WORD"], "HELLO WORLD")
    loss.backward()
    assert st["oracle_errors"] == 0 and st["expected_errors"] > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in net.trainable_parameters())


def test_word_edit_distance_and_tokenizer():
    assert word_edit_distance("A B C", "A X C") == 1
    assert word_edit_distance("", "A B") == 2
    assert normalize_text("Hey — it's 5 o’clock!") == "HEY IT'S 5 O'CLOCK"
    ids = Tokenizer().encode("hello world")
    assert ids.dtype == torch.long and len(ids) >= 2


def test_dataset_collate(tmp_path):
    rows = []
    for i, text in enumerate(["hello there", "send it now"]):
        s = preprocess([decode_wire_frame(m) for m in wire_frames(1.0 + i * 0.4, fps=25)])
        s.save(tmp_path / f"{i}.npz", text=text)
        rows.append({"path": str(tmp_path / f"{i}.npz"), "text": text, "frames": len(s.rois)})
    b = collate([UtteranceDataset(rows)[i] for i in range(2)])
    assert b["video"].shape == (2, 35, 88, 88) and b["lengths"].tolist() == [25, 35]
    assert (b["targets"][0] == -1).any() or (b["targets"][1] == -1).any()
