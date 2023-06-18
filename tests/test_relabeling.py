import pytest
import pandas as pd
from utils.relabeling import *

TAG_MAP_VALID = {
    'PER': ['DMN_LAST_NAME', 'DMN_FIRST_NAME', 'DMN_MIDDLE_NAME'],
    'LOC': ['DMN_COUNTRY', 'DMN_CITY'],
    'STREET': ['DMN_STREET', 'DMN_HOUSE'],
}
TAG_MAP_INVALID = {
    'PER': ['DMN_LAST_NAME', 'DMN_FIRST_NAME', 'DMN_MIDDLE_NAME'],
    'LOC': ['DMN_COUNTRY', 'DMN_CITY', 'DMN_STREET', 'DMN_HOUSE'],
    'STREET': ['DMN_STREET', 'DMN_HOUSE'],
}
sample = pd.DataFrame({
    'tokens': [
        ['Moscow', 'is', 'a', 'capital', 'of', 'Russia'],
        ['John', 'is', 'an', 'programmist'],
        ["Alex's", 'address', ':', 'Moscow', ',', 'Velikoy', 'Podedy', 'street', ',', '25'],
    ],
    'ner_tags': [
        ['U-DMN_CITY', 'O', 'O', 'O', 'O', 'U-DMN_COUNTRY'],
        ['U-DMN_FIRST_NAME', 'O', 'O', 'O'],
        ['U-DMN_FIRST_NAME', 'O', 'O', 'U-DMN_CITY', 'O', 'B-DMN_STREET', 'L-DMN_STREET', 'O', 'O', 'U-DMN_HOUSE'],
    ]
})


def test_biolu2bio():
    assert sample['ner_tags'].apply(biolu2bio).tolist() == [
        ['B-DMN_CITY', 'O', 'O', 'O', 'O', 'B-DMN_COUNTRY'],
        ['B-DMN_FIRST_NAME', 'O', 'O', 'O'],
        ['B-DMN_FIRST_NAME', 'O', 'O', 'B-DMN_CITY', 'O', 'B-DMN_STREET', 'I-DMN_STREET', 'O', 'O', 'B-DMN_HOUSE'],
    ]


def test_biolu2single_token():
    assert sample['ner_tags'].apply(biolu2single_token).tolist() == [
        ['DMN_CITY', 'O', 'O', 'O', 'O', 'DMN_COUNTRY'],
        ['DMN_FIRST_NAME', 'O', 'O', 'O'],
        ['DMN_FIRST_NAME', 'O', 'O', 'DMN_CITY', 'O', 'DMN_STREET', 'DMN_STREET', 'O', 'O', 'DMN_HOUSE'],
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
