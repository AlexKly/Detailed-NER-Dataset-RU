import pytest
import pandas as pd
from utils.relabeling import *

TAG_MAP_VALID = {
    'PER': ['LAST_NAME', 'FIRST_NAME', 'MIDDLE_NAME'],
    'LOC': ['COUNTRY', 'CITY'],
    'STREET': ['STREET', 'HOUSE'],
}
TAG_MAP_INVALID = {
    'PER': ['LAST_NAME', 'FIRST_NAME', 'MIDDLE_NAME'],
    'LOC': ['COUNTRY', 'CITY', 'STREET', 'HOUSE'],
    'STREET': ['STREET', 'HOUSE'],
}
sample = pd.DataFrame({
    'tokens': [
        ['Moscow', 'is', 'a', 'capital', 'of', 'Russia'],
        ['John', 'is', 'an', 'programmist'],
        ["Alex's", 'address', ':', 'Moscow', ',', 'Velikoy', 'Podedy', 'street', ',', '25'],
    ],
    'ner_tags': [
        ['U-CITY', 'O', 'O', 'O', 'O', 'U-COUNTRY'],
        ['U-FIRST_NAME', 'O', 'O', 'O'],
        ['U-FIRST_NAME', 'O', 'O', 'U-CITY', 'O', 'B-STREET', 'L-STREET', 'O', 'O', 'U-HOUSE'],
    ]
})


def test_biolu2bio():
    assert sample['ner_tags'].apply(biolu2bio).tolist() == [
        ['B-CITY', 'O', 'O', 'O', 'O', 'B-COUNTRY'],
        ['B-FIRST_NAME', 'O', 'O', 'O'],
        ['B-FIRST_NAME', 'O', 'O', 'B-CITY', 'O', 'B-STREET', 'I-STREET', 'O', 'O', 'B-HOUSE'],
    ]


def test_biolu2single_token():
    assert sample['ner_tags'].apply(biolu2single_token).tolist() == [
        ['CITY', 'O', 'O', 'O', 'O', 'COUNTRY'],
        ['FIRST_NAME', 'O', 'O', 'O'],
        ['FIRST_NAME', 'O', 'O', 'CITY', 'O', 'STREET', 'STREET', 'O', 'O', 'HOUSE'],
    ]


def test_detailed2default():
    assert sample['ner_tags'].apply(detailed2default).tolist() == [
        ['U-LOC', 'O', 'O', 'O', 'O', 'U-LOC'],
        ['U-PER', 'O', 'O', 'O'],
        ['U-PER', 'O', 'O', 'U-LOC', 'O', 'B-LOC', 'L-LOC', 'O', 'O', 'U-LOC'],
    ]


def test_detailed2custom_valid():
    assert sample['ner_tags'].apply(lambda x: detailed2custom(tags=x, tag_map=TAG_MAP_VALID)).tolist() == [
        ['U-LOC', 'O', 'O', 'O', 'O', 'U-LOC'],
        ['U-PER', 'O', 'O', 'O'],
        ['U-PER', 'O', 'O', 'U-LOC', 'O', 'B-STREET', 'L-STREET', 'O', 'O', 'U-STREET'],
    ]


def test_detailed2custom_invalid():
    # Has no changes because it has intersections in tag map:
    assert sample['ner_tags'].apply(
        lambda x: detailed2custom(tags=x, tag_map=TAG_MAP_INVALID)
    ).tolist() == sample['ner_tags'].tolist()
