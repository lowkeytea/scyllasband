import numpy as np
import pytest

from scyllasband.engine import Engine
from scyllasband.planner import normalize_request, parse_group_lines, plan_records, request_spec
from scyllasband.styles import Styles, is_style_request, parse_style

AXES = ("energy", "tension", "valence", "assertiveness")


def _control():
    def preset(e, t, v, a):
        return dict(energy=e, tension=t, valence=v, assertiveness=a)
    presets = {voice: {emotion: dict(less=preset(1.5, 1.8, 1.7, 2.1), normal=preset(1.8, 2.2, 1.7, 2.3),
                                     more=preset(2.1 + k, 2.7, 1.6, 2.5))
                       for k, emotion in enumerate(("neutral", "joy", "anger"))}
               for voice in ("ink", "max")}
    return dict(schema="scyllasband_styles_v1", emotions=["neutral", "joy", "anger"], intensities=["less", "normal", "more"],
                default=dict(emotion="neutral", intensity="normal"), axes=list(AXES),
                voice_rows=dict(base_voices=2, rule="row = voice_index + base_voices * emotions.index(emotion)"), presets=presets)


@pytest.mark.parametrize("value, expected", [
    ("joy", dict(emotion="joy", intensity=None)),
    ("more", dict(emotion=None, intensity="more")),
    ("joy,more", dict(emotion="joy", intensity="more")),
    ("emotion=anger,intensity=less", dict(emotion="anger", intensity="less")),
    ({"emotion": "Joy"}, dict(emotion="joy", intensity=None)),
    (None, dict(emotion=None, intensity=None)),
])
def test_parse_style(value, expected):
    assert parse_style(value) == expected


def test_style_requests_are_told_apart_from_numeric_delivery():
    assert is_style_request("joy,more") and is_style_request({"intensity": "less"})
    assert not is_style_request("energy=2.3,valence=2.2")
    assert not is_style_request("neutral=2") and not is_style_request(None)
    with pytest.raises(ValueError):
        parse_style("intensity=loud")
    with pytest.raises(ValueError):
        parse_style("joy,anger")          # one emotion only


def test_resolve_defaults_rows_and_tensors():
    styles = Styles(_control())
    default = styles.resolve("ink", None)
    assert (default["emotion"], default["intensity"]) == ("neutral", "normal")
    angry = styles.resolve("max", "anger,more")
    assert angry["values"]["energy"] == pytest.approx(4.1)
    assert styles.row(1, "neutral") == 1 and styles.row(1, "anger") == 1 + 2 * 2
    values, present = styles.tensors(default)
    assert present.all() and values[0, 4] == 0.0
    assert values[0, 0] == pytest.approx((1.8 - 2.0) / 4.0)
    with pytest.raises(ValueError):
        styles.resolve("ink", "sadness")   # not offered by this bundle


def test_planner_keeps_style_requests_and_chains_on_them():
    records = [dict(text="One. Two.", voice="ink", emotion="joy"), dict(text="Three.", voice="ink", emotion="joy", intensity="more"),
               dict(text="Four.", voice="ink")]
    plan = plan_records(records, resolve_language=lambda voice, language: "en_gb")
    assert plan.chunks[0].delivery == dict(emotion="joy", intensity=None, kind="style")
    assert len({c.chain_id for c in plan.chunks}) == 3
    assert plan.chunks[-1].delivery is None
    assert request_spec(plan.chunks[1].delivery) == "emotion=joy,intensity=default"
    with pytest.raises(ValueError):
        normalize_request(dict(delivery="energy=2", emotion="joy"))


def test_group_tags_take_styles():
    rows = parse_group_lines("[ink::anger,more] Put it down. [max:en_us:joy] We did it!", default_voice=None, default_language=None)
    assert rows[0]["voice"] == "ink" and rows[0]["delivery"] == dict(emotion="anger", intensity="more", kind="style")
    assert rows[1]["language"] == "en_us" and rows[1]["delivery"]["emotion"] == "joy"


class _StubEngine:
    voice_to_id = {"ink": 0, "max": 1}
    delivery_inputs = Engine.delivery_inputs
    voice_index = Engine.voice_index

    def __init__(self, styles):
        self.styles = styles


def test_engine_inputs_for_styles_and_older_bundles():
    styled = _StubEngine(Styles(_control()))
    values, present, row, resolved = styled.delivery_inputs("max", dict(emotion="joy", intensity="less", kind="style"))
    assert row == 1 + 2 and resolved["emotion"] == "joy" and present.all()
    _, _, row, resolved = styled.delivery_inputs("ink", None)
    assert row == 0 and resolved["emotion"] == "neutral"
    with pytest.raises(ValueError, match="emotion and an intensity"):
        styled.delivery_inputs("ink", dict(energy=2.3, tension=2.0, valence=2.0, assertiveness=2.0))
    older = _StubEngine(None)
    values, present, row, resolved = older.delivery_inputs("max", None)
    assert row == 1 and np.allclose(values[0, :4], 0.0)
    with pytest.raises(ValueError, match="no emotion styles"):
        older.delivery_inputs("ink", dict(emotion="joy", intensity=None, kind="style"))
