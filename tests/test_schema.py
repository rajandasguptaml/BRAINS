from brains.schema import LABELS, PatientRecord, format_outcomes, parse_outcomes, record_to_text


def _rec(**kw):
    base = dict(id="S1", age=76, gender="F", handedness="R", education_years=12, ses=3, mmse=22, cdr=1.0, etiv=1450,
                nwbv=0.68, diagnoses=[LABELS[2]])
    base.update(kw)
    return PatientRecord(**base)


def test_text_contains_fields_and_optional_outcome():
    r = _rec()
    t = record_to_text(r, include_outcome=False)
    assert "MMSE: 22" in t and "CDR: 1.0" in t and "eTIV (mL): 1450" in t and "Diagnostic Outcomes" not in t
    assert "Diagnostic Outcomes: Mild Alzheimer's Disease" in record_to_text(r, include_outcome=True)


def test_outcomes_roundtrip_all_case_types():
    for labels in ([LABELS[0]], [LABELS[1], LABELS[2]], [LABELS[2], LABELS[3], LABELS[4]]):
        s = format_outcomes(labels)
        assert s.startswith("Diagnostic Outcomes: [")
        assert parse_outcomes(s) == set(labels)


def test_parse_ignores_noise():
    assert parse_outcomes("blah [iii] and [v] and [x]") == {LABELS[2], LABELS[4]}
    assert parse_outcomes("nothing here") == set()


def test_case_type_and_validation():
    assert _rec().case_type == "single"
    assert _rec(diagnoses=[LABELS[0], LABELS[1]]).case_type == "double"
    try:
        _rec(diagnoses=["Not a label"])
        assert False, "expected ValueError"
    except ValueError:
        pass
