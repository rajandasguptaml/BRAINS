import numpy as np
import torch

from brains.encoder import CaseEncoder
from brains.fusion import CaseFusionLayer, pick_heads
from brains.retrieval import CaseMemory
from brains.synthetic import generate


def test_synthetic_generator_shapes():
    recs = generate(60, seed=1)
    assert len(recs) == 60
    assert {r.split for r in recs} <= {"train", "val", "test"}
    assert all(1 <= len(r.diagnoses) <= 3 for r in recs)
    assert all(0 <= r.mmse <= 30 for r in recs)


def test_memory_retrieves_similar_cases_and_excludes_self():
    recs = generate(80, seed=2)
    enc = CaseEncoder("hash")
    mem = CaseMemory(enc).build(recs)
    assert len(mem) == 80 and mem.embeddings.shape == (80, enc.dim)
    from brains.schema import record_to_text

    q = record_to_text(recs[0], include_outcome=False)
    got = mem.retrieve(q, top_k=5, top_n=50, exclude_ids={recs[0].id})
    assert 1 <= len(got) <= 5
    assert all(c.record.id != recs[0].id for c in got)
    assert all(got[i].score >= got[i + 1].score for i in range(len(got) - 1))
    assert mem.vectors(got).shape == (len(got), enc.dim)


def test_memory_save_load_roundtrip(tmp_path):
    recs = generate(30, seed=3)
    enc = CaseEncoder("hash")
    mem = CaseMemory(enc).build(recs)
    mem.save(tmp_path)
    mem2 = CaseMemory.load(tmp_path, enc)
    assert len(mem2) == 30
    np.testing.assert_allclose(mem.embeddings, mem2.embeddings)


def test_fusion_layer_shapes_masking_and_no_nan():
    torch.manual_seed(0)
    B, K, d_case, d_model = 3, 5, 16, 24
    layer = CaseFusionLayer(d_model, d_case, "auto")
    q = torch.randn(B, d_case)
    cases = torch.randn(B, K, d_case)
    mask = torch.ones(B, K, dtype=torch.bool)
    mask[1, 2:] = False  # partially masked row
    mask[2, :] = False  # fully masked row must not produce NaN
    out = layer(q, cases, mask)
    assert out.shape == (B, d_model)
    assert torch.isfinite(out).all()
    assert pick_heads(2) == 2 and pick_heads(576) == 8 and pick_heads(12) == 4 and pick_heads(7) == 1
