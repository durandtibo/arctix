from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import pytest
from coola import objects_are_equal
from iden.io import save_json
from polars.testing import assert_frame_equal

from arctix.dataset.ego4d import (
    NUM_NOUNS,
    NUM_VERBS,
    Column,
    MetadataKeys,
    fetch_data,
    group_by_sequence,
    load_data,
    load_event_data,
    load_noun_vocab,
    load_taxonomy_vocab,
    load_verb_vocab,
    prepare_data,
    to_array,
    to_list,
)
from arctix.utils.vocab import Vocabulary

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("ego4d")
    create_train_annotations(path)
    create_taxonomy_file(path)
    return path


def create_train_annotations(path: Path) -> None:
    save_json(
        {
            "version": "v2",
            "date": "230208",
            "description": "blabla",
            "split": "train",
            "clips": [
                {
                    "action_clip_end_frame": 47,
                    "action_clip_end_sec": 4.7,
                    "action_clip_start_frame": 23,
                    "action_clip_start_sec": 2.3,
                    "action_idx": 0,
                    "clip_uid": "clip1",
                    "noun": "noun2",
                    "noun_label": 2,
                    "other": "blabla...",
                    "verb": "verb4",
                    "verb_label": 4,
                    "video_uid": "video1",
                },
                {
                    "action_clip_end_frame": 82,
                    "action_clip_end_sec": 8.2,
                    "action_clip_start_frame": 39,
                    "action_clip_start_sec": 3.9,
                    "action_idx": 1,
                    "clip_uid": "clip1",
                    "noun": "noun3",
                    "noun_label": 3,
                    "other": "blabla...",
                    "verb": "verb2",
                    "verb_label": 2,
                    "video_uid": "video1",
                },
                {
                    "action_clip_end_frame": 102,
                    "action_clip_end_sec": 10.2,
                    "action_clip_start_frame": 74,
                    "action_clip_start_sec": 7.4,
                    "action_idx": 2,
                    "clip_uid": "clip1",
                    "noun": "noun1",
                    "noun_label": 1,
                    "other": "blabla...",
                    "verb": "verb1",
                    "verb_label": 1,
                    "video_uid": "video1",
                },
                {
                    "action_clip_end_frame": 74,
                    "action_clip_end_sec": 7.4,
                    "action_clip_start_frame": 12,
                    "action_clip_start_sec": 1.2,
                    "action_idx": 0,
                    "clip_uid": "clip2",
                    "noun": "noun1",
                    "noun_label": 1,
                    "other": "blabla...",
                    "verb": "verb1",
                    "verb_label": 1,
                    "video_uid": "video2",
                },
                {
                    "action_clip_end_frame": 142,
                    "action_clip_end_sec": 14.2,
                    "action_clip_start_frame": 82,
                    "action_clip_start_sec": 8.2,
                    "action_idx": 1,
                    "clip_uid": "clip2",
                    "noun": "noun2",
                    "noun_label": 2,
                    "other": "blabla...",
                    "verb": "verb2",
                    "verb_label": 2,
                    "video_uid": "video2",
                },
            ],
        },
        path.joinpath("ego4d_data/v2/annotations/fho_lta_train.json"),
    )


def create_taxonomy_file(path: Path) -> None:
    save_json(
        {
            "nouns": [f"noun{i:03}" for i in range(NUM_NOUNS)],
            "verbs": [f"verb{i:03}" for i in range(NUM_VERBS)],
        },
        path.joinpath("ego4d_data/v2/annotations/fho_lta_taxonomy.json"),
    )


@pytest.fixture
def vocab_noun() -> Vocabulary:
    return Vocabulary(Counter({f"noun{i:03}": 1 for i in range(NUM_NOUNS)}))


@pytest.fixture
def vocab_verb() -> Vocabulary:
    return Vocabulary(Counter({f"verb{i:03}": 1 for i in range(NUM_VERBS)}))


@pytest.fixture
def data_raw() -> pl.DataFrame:
    return pl.DataFrame(
        {
            Column.ACTION_END_FRAME: [47, 82, 102, 74, 142],
            Column.ACTION_END_SEC: [4.7, 8.2, 10.2, 7.4, 14.2],
            Column.ACTION_START_FRAME: [23, 39, 74, 12, 82],
            Column.ACTION_START_SEC: [2.3, 3.9, 7.4, 1.2, 8.2],
            Column.ACTION_INDEX: [0, 1, 2, 0, 1],
            Column.CLIP_ID: ["clip1", "clip1", "clip1", "clip2", "clip2"],
            Column.NOUN: ["noun2", "noun3", "noun1", "noun1", "noun2"],
            Column.NOUN_ID: [2, 3, 1, 1, 2],
            Column.SPLIT: ["train", "train", "train", "train", "train"],
            Column.VERB: ["verb4", "verb2", "verb1", "verb1", "verb2"],
            Column.VERB_ID: [4, 2, 1, 1, 2],
            Column.VIDEO_ID: ["video1", "video1", "video1", "video2", "video2"],
        },
        schema={
            Column.ACTION_END_FRAME: pl.Int64,
            Column.ACTION_END_SEC: pl.Float64,
            Column.ACTION_START_FRAME: pl.Int64,
            Column.ACTION_START_SEC: pl.Float64,
            Column.ACTION_INDEX: pl.Int64,
            Column.CLIP_ID: pl.String,
            Column.NOUN: pl.String,
            Column.NOUN_ID: pl.Int64,
            Column.SPLIT: pl.String,
            Column.VERB: pl.String,
            Column.VERB_ID: pl.Int64,
            Column.VIDEO_ID: pl.String,
        },
    )


@pytest.fixture
def data_prepared() -> pl.DataFrame:
    return pl.DataFrame(
        {
            Column.ACTION_END_FRAME: [47, 82, 102, 74, 142],
            Column.ACTION_END_SEC: [4.7, 8.2, 10.2, 7.4, 14.2],
            Column.ACTION_START_FRAME: [23, 39, 74, 12, 82],
            Column.ACTION_START_SEC: [2.3, 3.9, 7.4, 1.2, 8.2],
            Column.ACTION_START_SEC_DIFF: [0.0, 1.6, 3.5, 0.0, 7.0],
            Column.ACTION_INDEX: [0, 1, 2, 0, 1],
            Column.CLIP_ID: ["clip1", "clip1", "clip1", "clip2", "clip2"],
            Column.NOUN: ["noun2", "noun3", "noun1", "noun1", "noun2"],
            Column.NOUN_ID: [2, 3, 1, 1, 2],
            Column.SPLIT: ["train", "train", "train", "train", "train"],
            Column.VERB: ["verb4", "verb2", "verb1", "verb1", "verb2"],
            Column.VERB_ID: [4, 2, 1, 1, 2],
            Column.VIDEO_ID: ["video1", "video1", "video1", "video2", "video2"],
        },
        schema={
            Column.ACTION_END_FRAME: pl.Int64,
            Column.ACTION_END_SEC: pl.Float64,
            Column.ACTION_START_FRAME: pl.Int64,
            Column.ACTION_START_SEC: pl.Float64,
            Column.ACTION_START_SEC_DIFF: pl.Float64,
            Column.ACTION_INDEX: pl.Int64,
            Column.CLIP_ID: pl.String,
            Column.NOUN: pl.String,
            Column.NOUN_ID: pl.Int64,
            Column.SPLIT: pl.String,
            Column.VERB: pl.String,
            Column.VERB_ID: pl.Int64,
            Column.VIDEO_ID: pl.String,
        },
    )


################################
#     Tests for fetch_data     #
################################


def test_fetch_data(
    data_dir: Path, data_raw: pl.DataFrame, vocab_noun: Vocabulary, vocab_verb: Vocabulary
) -> None:
    data, metadata = fetch_data(data_dir, split="train")
    assert_frame_equal(data, data_raw)
    assert objects_are_equal(
        metadata, {MetadataKeys.VOCAB_NOUN: vocab_noun, MetadataKeys.VOCAB_VERB: vocab_verb}
    )


###############################
#     Tests for load_data     #
###############################


def test_load_data(
    data_dir: Path, data_raw: pl.DataFrame, vocab_noun: Vocabulary, vocab_verb: Vocabulary
) -> None:
    data, metadata = load_data(data_dir, split="train")
    assert_frame_equal(data, data_raw)
    assert objects_are_equal(
        metadata, {MetadataKeys.VOCAB_NOUN: vocab_noun, MetadataKeys.VOCAB_VERB: vocab_verb}
    )


#####################################
#     Tests for load_event_data     #
#####################################


def test_load_event_data(data_dir: Path, data_raw: pl.DataFrame) -> None:
    assert_frame_equal(load_event_data(data_dir, split="train"), data_raw)


#####################################
#     Tests for load_noun_vocab     #
#####################################


def test_load_noun_vocab(data_dir: Path, vocab_noun: Vocabulary) -> None:
    assert load_noun_vocab(data_dir).equal(vocab_noun)


#####################################
#     Tests for load_verb_vocab     #
#####################################


def test_load_verb_vocab(data_dir: Path, vocab_verb: Vocabulary) -> None:
    assert load_verb_vocab(data_dir).equal(vocab_verb)


#########################################
#     Tests for load_taxonomy_vocab     #
#########################################


def test_load_taxonomy_vocab_nouns(data_dir: Path, vocab_noun: Vocabulary) -> None:
    assert load_taxonomy_vocab(data_dir, name="nouns").equal(vocab_noun)


def test_load_taxonomy_vocab_verbs(data_dir: Path, vocab_verb: Vocabulary) -> None:
    assert load_taxonomy_vocab(data_dir, name="verbs").equal(vocab_verb)


def test_load_taxonomy_vocab_expected_size(data_dir: Path) -> None:
    assert load_taxonomy_vocab(data_dir, name="nouns", expected_size=NUM_NOUNS).equal(
        Vocabulary(Counter({f"noun{i:03}": 1 for i in range(NUM_NOUNS)}))
    )


def test_load_taxonomy_vocab_expected_size_incorrect(data_dir: Path) -> None:
    with pytest.raises(RuntimeError, match="Expected 1 nouns but received"):
        load_taxonomy_vocab(data_dir, name="nouns", expected_size=1)


def test_load_taxonomy_vocab_incorrect_name(data_dir: Path) -> None:
    with pytest.raises(RuntimeError, match="Incorrect taxonomy name:"):
        load_taxonomy_vocab(data_dir, name="incorrect")


##################################
#     Tests for prepare_data     #
##################################


def test_prepare_data(
    data_raw: pl.DataFrame,
    data_prepared: pl.DataFrame,
    vocab_noun: Vocabulary,
    vocab_verb: Vocabulary,
) -> None:
    data, metadata = prepare_data(
        data_raw,
        metadata={MetadataKeys.VOCAB_NOUN: vocab_noun, MetadataKeys.VOCAB_VERB: vocab_verb},
    )
    assert_frame_equal(data, data_prepared)
    assert objects_are_equal(
        metadata, {MetadataKeys.VOCAB_NOUN: vocab_noun, MetadataKeys.VOCAB_VERB: vocab_verb}
    )


#######################################
#     Tests for group_by_sequence     #
#######################################


def test_group_by_sequence_clip_id(data_prepared: pl.DataFrame) -> None:
    data = group_by_sequence(data_prepared)
    assert_frame_equal(
        data,
        pl.DataFrame(
            {
                Column.ACTION_END_FRAME: [[47, 82, 102], [74, 142]],
                Column.ACTION_END_SEC: [[4.7, 8.2, 10.2], [7.4, 14.2]],
                Column.ACTION_START_FRAME: [[23, 39, 74], [12, 82]],
                Column.ACTION_START_SEC: [[2.3, 3.9, 7.4], [1.2, 8.2]],
                Column.ACTION_START_SEC_DIFF: [[0.0, 1.6, 3.5], [0.0, 7.0]],
                Column.CLIP_ID: ["clip1", "clip2"],
                Column.NOUN: [["noun2", "noun3", "noun1"], ["noun1", "noun2"]],
                Column.NOUN_ID: [[2, 3, 1], [1, 2]],
                Column.SEQUENCE_LENGTH: [3, 2],
                Column.SPLIT: ["train", "train"],
                Column.VERB: [["verb4", "verb2", "verb1"], ["verb1", "verb2"]],
                Column.VERB_ID: [[4, 2, 1], [1, 2]],
            },
            schema={
                Column.ACTION_END_FRAME: pl.List(pl.Int64),
                Column.ACTION_END_SEC: pl.List(pl.Float64),
                Column.ACTION_START_FRAME: pl.List(pl.Int64),
                Column.ACTION_START_SEC: pl.List(pl.Float64),
                Column.ACTION_START_SEC_DIFF: pl.List(pl.Float64),
                Column.CLIP_ID: pl.String,
                Column.NOUN: pl.List(pl.String),
                Column.NOUN_ID: pl.List(pl.Int64),
                Column.SEQUENCE_LENGTH: pl.Int64,
                Column.SPLIT: pl.String,
                Column.VERB: pl.List(pl.String),
                Column.VERB_ID: pl.List(pl.Int64),
            },
        ),
    )


def test_group_by_sequence_video_id(data_prepared: pl.DataFrame) -> None:
    data = group_by_sequence(data_prepared, group_col=Column.VIDEO_ID)
    assert_frame_equal(
        data,
        pl.DataFrame(
            {
                Column.ACTION_END_FRAME: [[47, 82, 102], [74, 142]],
                Column.ACTION_END_SEC: [[4.7, 8.2, 10.2], [7.4, 14.2]],
                Column.ACTION_START_FRAME: [[23, 39, 74], [12, 82]],
                Column.ACTION_START_SEC: [[2.3, 3.9, 7.4], [1.2, 8.2]],
                Column.ACTION_START_SEC_DIFF: [[0.0, 1.6, 3.5], [0.0, 7.0]],
                Column.NOUN: [["noun2", "noun3", "noun1"], ["noun1", "noun2"]],
                Column.NOUN_ID: [[2, 3, 1], [1, 2]],
                Column.SEQUENCE_LENGTH: [3, 2],
                Column.SPLIT: ["train", "train"],
                Column.VERB: [["verb4", "verb2", "verb1"], ["verb1", "verb2"]],
                Column.VERB_ID: [[4, 2, 1], [1, 2]],
                Column.VIDEO_ID: ["video1", "video2"],
            },
            schema={
                Column.ACTION_END_FRAME: pl.List(pl.Int64),
                Column.ACTION_END_SEC: pl.List(pl.Float64),
                Column.ACTION_START_FRAME: pl.List(pl.Int64),
                Column.ACTION_START_SEC: pl.List(pl.Float64),
                Column.ACTION_START_SEC_DIFF: pl.List(pl.Float64),
                Column.NOUN: pl.List(pl.String),
                Column.NOUN_ID: pl.List(pl.Int64),
                Column.SEQUENCE_LENGTH: pl.Int64,
                Column.SPLIT: pl.String,
                Column.VERB: pl.List(pl.String),
                Column.VERB_ID: pl.List(pl.Int64),
                Column.VIDEO_ID: pl.String,
            },
        ),
    )


##############################
#     Tests for to_array     #
##############################


def test_to_array_clip_id(data_prepared: pl.DataFrame) -> None:
    mask = np.array([[False, False, False], [False, False, True]])
    assert objects_are_equal(
        to_array(data_prepared),
        {
            Column.ACTION_END_FRAME: np.ma.masked_array(
                data=np.array([[47, 82, 102], [74, 142, -1]], dtype=np.int64), mask=mask
            ),
            Column.ACTION_END_SEC: np.ma.masked_array(
                data=np.array([[4.7, 8.2, 10.2], [7.4, 14.2, -1.0]], dtype=np.float64), mask=mask
            ),
            Column.ACTION_START_FRAME: np.ma.masked_array(
                data=np.array([[23, 39, 74], [12, 82, -1]], dtype=np.int64), mask=mask
            ),
            Column.ACTION_START_SEC: np.ma.masked_array(
                data=np.array([[2.3, 3.9, 7.4], [1.2, 8.2, -1.0]], dtype=np.float64), mask=mask
            ),
            Column.ACTION_START_SEC_DIFF: np.ma.masked_array(
                data=np.array([[0.0, 1.6, 3.5], [0.0, 7.0, -1.0]], dtype=np.float64), mask=mask
            ),
            Column.CLIP_ID: np.array(["clip1", "clip2"]),
            Column.NOUN: np.ma.masked_array(
                data=np.array([["noun2", "noun3", "noun1"], ["noun1", "noun2", "N/A"]]),
                mask=mask,
            ),
            Column.NOUN_ID: np.ma.masked_array(
                data=np.array([[2, 3, 1], [1, 2, -1]], dtype=np.int64), mask=mask
            ),
            Column.SEQUENCE_LENGTH: np.array([3, 2], dtype=np.int64),
            Column.SPLIT: np.array(["train", "train"]),
            Column.VERB: np.ma.masked_array(
                data=np.array([["verb4", "verb2", "verb1"], ["verb1", "verb2", "N/A"]]),
                mask=mask,
            ),
            Column.VERB_ID: np.ma.masked_array(
                data=np.array([[4, 2, 1], [1, 2, -1]], dtype=np.int64), mask=mask
            ),
        },
    )


def test_to_array_video_id(data_prepared: pl.DataFrame) -> None:
    mask = np.array([[False, False, False], [False, False, True]])
    assert objects_are_equal(
        to_array(data_prepared, group_col=Column.VIDEO_ID),
        {
            Column.ACTION_END_FRAME: np.ma.masked_array(
                data=np.array([[47, 82, 102], [74, 142, -1]], dtype=np.int64), mask=mask
            ),
            Column.ACTION_END_SEC: np.ma.masked_array(
                data=np.array([[4.7, 8.2, 10.2], [7.4, 14.2, -1.0]], dtype=np.float64), mask=mask
            ),
            Column.ACTION_START_FRAME: np.ma.masked_array(
                data=np.array([[23, 39, 74], [12, 82, -1]], dtype=np.int64), mask=mask
            ),
            Column.ACTION_START_SEC: np.ma.masked_array(
                data=np.array([[2.3, 3.9, 7.4], [1.2, 8.2, -1.0]], dtype=np.float64), mask=mask
            ),
            Column.ACTION_START_SEC_DIFF: np.ma.masked_array(
                data=np.array([[0.0, 1.6, 3.5], [0.0, 7.0, -1.0]], dtype=np.float64), mask=mask
            ),
            Column.NOUN: np.ma.masked_array(
                data=np.array([["noun2", "noun3", "noun1"], ["noun1", "noun2", "N/A"]]),
                mask=mask,
            ),
            Column.NOUN_ID: np.ma.masked_array(
                data=np.array([[2, 3, 1], [1, 2, -1]], dtype=np.int64), mask=mask
            ),
            Column.SEQUENCE_LENGTH: np.array([3, 2], dtype=np.int64),
            Column.SPLIT: np.array(["train", "train"]),
            Column.VERB: np.ma.masked_array(
                data=np.array([["verb4", "verb2", "verb1"], ["verb1", "verb2", "N/A"]]),
                mask=mask,
            ),
            Column.VERB_ID: np.ma.masked_array(
                data=np.array([[4, 2, 1], [1, 2, -1]], dtype=np.int64), mask=mask
            ),
            Column.VIDEO_ID: np.array(["video1", "video2"]),
        },
    )


#############################
#     Tests for to_list     #
#############################


def test_to_list_clip_id(data_prepared: pl.DataFrame) -> None:
    assert objects_are_equal(
        to_list(data_prepared),
        {
            Column.ACTION_END_FRAME: [[47, 82, 102], [74, 142]],
            Column.ACTION_END_SEC: [[4.7, 8.2, 10.2], [7.4, 14.2]],
            Column.ACTION_START_FRAME: [[23, 39, 74], [12, 82]],
            Column.ACTION_START_SEC: [[2.3, 3.9, 7.4], [1.2, 8.2]],
            Column.ACTION_START_SEC_DIFF: [[0.0, 1.6, 3.5], [0.0, 7.0]],
            Column.CLIP_ID: ["clip1", "clip2"],
            Column.NOUN: [["noun2", "noun3", "noun1"], ["noun1", "noun2"]],
            Column.NOUN_ID: [[2, 3, 1], [1, 2]],
            Column.SEQUENCE_LENGTH: [3, 2],
            Column.SPLIT: ["train", "train"],
            Column.VERB: [["verb4", "verb2", "verb1"], ["verb1", "verb2"]],
            Column.VERB_ID: [[4, 2, 1], [1, 2]],
        },
    )


def test_to_list_video_id(data_prepared: pl.DataFrame) -> None:
    assert objects_are_equal(
        to_list(data_prepared, group_col=Column.VIDEO_ID),
        {
            Column.ACTION_END_FRAME: [[47, 82, 102], [74, 142]],
            Column.ACTION_END_SEC: [[4.7, 8.2, 10.2], [7.4, 14.2]],
            Column.ACTION_START_FRAME: [[23, 39, 74], [12, 82]],
            Column.ACTION_START_SEC: [[2.3, 3.9, 7.4], [1.2, 8.2]],
            Column.ACTION_START_SEC_DIFF: [[0.0, 1.6, 3.5], [0.0, 7.0]],
            Column.NOUN: [["noun2", "noun3", "noun1"], ["noun1", "noun2"]],
            Column.NOUN_ID: [[2, 3, 1], [1, 2]],
            Column.SEQUENCE_LENGTH: [3, 2],
            Column.SPLIT: ["train", "train"],
            Column.VERB: [["verb4", "verb2", "verb1"], ["verb1", "verb2"]],
            Column.VERB_ID: [[4, 2, 1], [1, 2]],
            Column.VIDEO_ID: ["video1", "video2"],
        },
    )
