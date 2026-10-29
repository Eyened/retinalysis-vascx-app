"""Run with pytest in a Python environment containing retinalysis-vascx."""

import base64
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture(scope="module")
def helpers():
    source = (Path(__file__).parents[1] / "src/lib/workers/feature.worker.ts").read_text()
    python_source = source.split("const BROWSER_HELPERS_PY = String.raw`", 1)[1].rsplit("`;", 1)[0]
    namespace = {}
    exec(python_source, namespace)
    return namespace


@pytest.mark.parametrize("feature_set_name", ["od_centered", "macula_centered"])
def test_manifest_lists_individual_outputs(helpers, feature_set_name):
    feature_set = helpers["_feature_set"](feature_set_name)
    expected = helpers["Retina"].make_feature_display_names(feature_set, naming="resolved")
    manifest = next(item for item in helpers["feature_set_manifest"]() if item["name"] == feature_set_name)
    assert {item["key"]: item["name"] for item in manifest["biomarkers"]} == expected


@pytest.mark.parametrize("feature_class", ["BackgroundLab", "Bifurcation", "ODMorphology", "Caliber"])
def test_output_components_plot_the_parent_feature(helpers, monkeypatch, feature_class):
    feature_set = helpers["_feature_set"]("od_centered")
    layers = {name: SimpleNamespace(name=name) for name in ("arteries", "veins", "vessels")}
    retina = SimpleNamespace(get_layers=lambda _: list(layers.values()))
    bindings = [item for item in feature_set.names().by_name.values()
                if type(item.feature).__name__ == feature_class]
    assert bindings
    for binding in bindings:
        calls = []

        def plot(ax, **kwargs):
            calls.append(kwargs)
            ax.plot([0, 1], [0, 1])

        monkeypatch.setattr(binding.feature, "plot", plot)
        uri = helpers["plot_feature_data_uri"](retina, "od_centered", binding.name)
        expected = {"retina": retina} if binding.target == "retina" else {"layer": layers[binding.target]}
        assert calls == [expected]
        assert uri.startswith("data:image/png;base64,")
        assert base64.b64decode(uri.split(",", 1)[1]).startswith(b"\x89PNG\r\n\x1a\n")
        assert not helpers["plt"].get_fignums()


def test_plot_failure_closes_figure(helpers, monkeypatch):
    feature_set = helpers["_feature_set"]("od_centered")
    binding = next(item for item in feature_set.names().by_name.values()
                   if type(item.feature).__name__ == "BackgroundLab")

    def fail(**kwargs):
        raise ValueError("Cannot plot this image")

    monkeypatch.setattr(binding.feature, "plot", fail)
    with pytest.raises(ValueError, match="Cannot plot this image"):
        helpers["plot_feature_data_uri"](object(), "od_centered", binding.name)
    assert not helpers["plt"].get_fignums()
