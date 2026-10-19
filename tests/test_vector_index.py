from db import vector_index_definition


def test_vector_index_contains_all_retrieval_modalities_and_source_filter():
    definition = vector_index_definition()
    vector_fields = {
        field["path"]
        for field in definition["fields"]
        if field["type"] == "vector"
    }

    assert vector_fields == {
        "embedding",
        "visual_embedding",
        "transcript_embedding",
        "ocr_embedding",
    }
    assert {"type": "filter", "path": "source"} in definition["fields"]
