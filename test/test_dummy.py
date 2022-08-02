from os import path


def test_dummy():
    assert path.isfile("sample_data/dummy.txt")