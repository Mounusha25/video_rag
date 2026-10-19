from time_stamp_grouping import get_timestamps


def test_groups_nearby_timestamps_and_adds_context():
    assert get_timestamps([5, 10, 20, 50], video_length=60) == [
        (3, 24),
        (48, 54),
    ]


def test_deduplicates_sorts_and_clamps_segments():
    assert get_timestamps([29, 1, 1], video_length=30, gap_threshold=5) == [
        (0, 5),
        (27, 30),
    ]


def test_empty_input_has_no_segments():
    assert get_timestamps([], video_length=100) == []
