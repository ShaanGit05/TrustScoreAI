# tests/test_data_loader.py
import json
import tempfile
from pathlib import Path
from src.data_loader import load_json, validate_dataset, prepare_prompts, save_processed

SAMPLE = [
    {
        "id": "t1",
        "prompt": "Describe {occupation}.",
        "baseline": "Occupations should be respected.",
        "protected_attribute": "occupation",
        "protected_value": "taxi driver",
        "metadata": {"occupation": "taxi driver"}
    }
]


def test_load_and_prepare(tmp_path):
    p = tmp_path / "prompts.json"
    p.write_text(json.dumps(SAMPLE, ensure_ascii=False))
    records = load_json(str(p))
    # validate should not raise
    validate_dataset(records)
    processed = prepare_prompts(records)
    assert processed[0]["id"] == "t1"
    assert "prompt_text" in processed[0]
    outp = tmp_path / "processed.json"
    save_processed(processed, str(outp))
    assert outp.exists()
