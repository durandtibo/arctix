from __future__ import annotations

import datetime
from collections import Counter
from typing import TYPE_CHECKING
from unittest.mock import Mock, patch
from zipfile import ZipFile

import polars as pl
import pytest
from coola import objects_are_equal
from iden.io import save_text
from polars.testing import assert_frame_equal

from arctix.dataset.epic_kitchen_100 import (
    ANNOTATION_FILENAMES,
    ANNOTATION_URL,
    NUM_NOUNS,
    NUM_VERBS,
    Column,
    download_data,
    fetch_data,
    is_annotation_path_ready,
    load_data,
    load_event_data,
    load_noun_vocab,
    load_verb_vocab,
    prepare_data,
)
from arctix.utils.vocab import Vocabulary

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(scope="module")
def data_zip_file(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("data").joinpath("epic-kitchens-100.zip.partial")
    path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(path, "w") as zfile:
        for filename in ANNOTATION_FILENAMES:
            zfile.writestr(f"epic-kitchens-100-annotations-master/{filename}", "")
    return path


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("data")


@pytest.fixture(scope="module", autouse=True)
def data_file(data_dir: Path) -> Path:
    path = data_dir.joinpath("EPIC_100_train.csv")
    save_text(
        "narration_id,participant_id,video_id,narration_timestamp,start_timestamp,"
        "stop_timestamp,start_frame,stop_frame,narration,verb,verb_class,noun,noun_class,"
        "all_nouns,all_noun_classes\n"
        "P01_01_2,P01,P01_01,00:00:05.349,00:00:06.98,00:00:09.49,418,569,close door,close,4,"
        "door,3,['door'],[3]\n"
        "P01_01_0,P01,P01_01,00:00:01.089,00:00:00.14,00:00:03.37,8,202,open door,open,3,door,3,"
        "['door'],[3]\n"
        "P01_01_1,P01,P01_01,00:00:02.629,00:00:04.37,00:00:06.17,262,370,turn on light,turn-on,6,"
        "light,114,['light'],[114]\n",
        path,
    )
    return path


@pytest.fixture(scope="module")
def empty_data_file(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("empty").joinpath("EPIC_100_train.csv")
    save_text(
        "narration_id,participant_id,video_id,narration_timestamp,start_timestamp,"
        "stop_timestamp,start_frame,stop_frame,narration,verb,verb_class,noun,noun_class,"
        "all_nouns,all_noun_classes\n",
        path,
    )
    return path


@pytest.fixture(scope="module", autouse=True)
def noun_file(data_dir: Path) -> Path:
    path = data_dir.joinpath("EPIC_100_noun_classes.csv")
    lines = [f"{i},{i}n{i}" for i in range(NUM_NOUNS)]
    lines.insert(0, "id,key")
    save_text("\n".join(lines), path)
    return path


@pytest.fixture(scope="module")
def empty_noun_file(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("empty").joinpath("EPIC_100_noun_classes.csv")
    save_text("id,key\n", path)
    return path


@pytest.fixture(scope="module", autouse=True)
def verb_file(data_dir: Path) -> Path:
    path = data_dir.joinpath("EPIC_100_verb_classes.csv")
    lines = [f"{i},{i}v{i}" for i in range(NUM_VERBS)]
    lines.insert(0, "id,key")
    save_text("\n".join(lines), path)
    return path


@pytest.fixture(scope="module")
def empty_verb_file(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("empty").joinpath("EPIC_100_verb_classes.csv")
    save_text("id,key\n", path)
    return path


@pytest.fixture()
def data_raw() -> pl.DataFrame:
    return pl.DataFrame(
        {
            Column.ALL_NOUN_IDS: [[3], [114], [3]],
            Column.ALL_NOUNS: [["door"], ["light"], ["door"]],
            Column.NARRATION: ["open door", "turn on light", "close door"],
            Column.NARRATION_ID: ["P01_01_0", "P01_01_1", "P01_01_2"],
            Column.NARRATION_TIMESTAMP: [
                datetime.time(0, 0, 1, 89000),
                datetime.time(0, 0, 2, 629000),
                datetime.time(0, 0, 5, 349000),
            ],
            Column.NOUN: ["door", "light", "door"],
            Column.NOUN_ID: [3, 114, 3],
            Column.PARTICIPANT_ID: ["P01", "P01", "P01"],
            Column.START_FRAME: [8, 262, 418],
            Column.START_TIMESTAMP: [
                datetime.time(0, 0, 0, 140000),
                datetime.time(0, 0, 4, 370000),
                datetime.time(0, 0, 6, 980000),
            ],
            Column.STOP_FRAME: [202, 370, 569],
            Column.STOP_TIMESTAMP: [
                datetime.time(0, 0, 3, 370000),
                datetime.time(0, 0, 6, 170000),
                datetime.time(0, 0, 9, 490000),
            ],
            Column.VERB: ["open", "turn-on", "close"],
            Column.VERB_ID: [3, 6, 4],
            Column.VIDEO_ID: ["P01_01", "P01_01", "P01_01"],
        },
        schema={
            Column.ALL_NOUN_IDS: pl.List(pl.Int64),
            Column.ALL_NOUNS: pl.List(pl.String),
            Column.NARRATION: pl.String,
            Column.NARRATION_ID: pl.String,
            Column.NARRATION_TIMESTAMP: pl.Time,
            Column.NOUN: pl.String,
            Column.NOUN_ID: pl.Int64,
            Column.PARTICIPANT_ID: pl.String,
            Column.START_FRAME: pl.Int64,
            Column.START_TIMESTAMP: pl.Time,
            Column.STOP_FRAME: pl.Int64,
            Column.STOP_TIMESTAMP: pl.Time,
            Column.VERB: pl.String,
            Column.VERB_ID: pl.Int64,
            Column.VIDEO_ID: pl.String,
        },
    )


@pytest.fixture()
def data_prepared() -> pl.DataFrame:
    return pl.DataFrame(
        {
            Column.ALL_NOUN_IDS: [[3], [114], [3]],
            Column.ALL_NOUNS: [["door"], ["light"], ["door"]],
            Column.NARRATION: ["open door", "turn on light", "close door"],
            Column.NARRATION_ID: ["P01_01_0", "P01_01_1", "P01_01_2"],
            Column.NARRATION_TIMESTAMP: [
                datetime.time(0, 0, 1, 89000),
                datetime.time(0, 0, 2, 629000),
                datetime.time(0, 0, 5, 349000),
            ],
            Column.NOUN: ["door", "light", "door"],
            Column.NOUN_ID: [3, 114, 3],
            Column.PARTICIPANT_ID: ["P01", "P01", "P01"],
            Column.START_FRAME: [8, 262, 418],
            Column.START_TIME_SECOND: [0.14, 4.37, 6.98],
            Column.START_TIMESTAMP: [
                datetime.time(0, 0, 0, 140000),
                datetime.time(0, 0, 4, 370000),
                datetime.time(0, 0, 6, 980000),
            ],
            Column.STOP_FRAME: [202, 370, 569],
            Column.STOP_TIME_SECOND: [3.37, 6.17, 9.49],
            Column.STOP_TIMESTAMP: [
                datetime.time(0, 0, 3, 370000),
                datetime.time(0, 0, 6, 170000),
                datetime.time(0, 0, 9, 490000),
            ],
            Column.VERB: ["open", "turn-on", "close"],
            Column.VERB_ID: [3, 6, 4],
            Column.VIDEO_ID: ["P01_01", "P01_01", "P01_01"],
        },
        schema={
            Column.ALL_NOUN_IDS: pl.List(pl.Int64),
            Column.ALL_NOUNS: pl.List(pl.String),
            Column.NARRATION: pl.String,
            Column.NARRATION_ID: pl.String,
            Column.NARRATION_TIMESTAMP: pl.Time,
            Column.NOUN: pl.String,
            Column.NOUN_ID: pl.Int64,
            Column.PARTICIPANT_ID: pl.String,
            Column.START_FRAME: pl.Int64,
            Column.START_TIME_SECOND: pl.Float32,
            Column.START_TIMESTAMP: pl.Time,
            Column.STOP_FRAME: pl.Int64,
            Column.STOP_TIME_SECOND: pl.Float32,
            Column.STOP_TIMESTAMP: pl.Time,
            Column.VERB: pl.String,
            Column.VERB_ID: pl.Int64,
            Column.VIDEO_ID: pl.String,
        },
    )


@pytest.fixture()
def noun_vocab() -> Vocabulary:
    return Vocabulary(Counter({f"{i}n{i}": 1 for i in range(NUM_NOUNS)}))


@pytest.fixture()
def verb_vocab() -> Vocabulary:
    return Vocabulary(Counter({f"{i}v{i}": 1 for i in range(NUM_VERBS)}))


################################
#     Tests for fetch_data     #
################################


def test_fetch_data(
    data_dir: Path,
    data_raw: pl.DataFrame,
    noun_vocab: Vocabulary,
    verb_vocab: Vocabulary,
) -> None:
    with patch("arctix.dataset.epic_kitchen_100.download_data") as download_mock:
        data, metadata = fetch_data(data_dir, split="train")
        download_mock.assert_called_once_with(data_dir, False)
        assert_frame_equal(data, data_raw)
        assert objects_are_equal(metadata, {"noun_vocab": noun_vocab, "verb_vocab": verb_vocab})


###################################
#     Tests for download_data     #
###################################


@pytest.mark.parametrize("force_download", [True, False])
def test_download_data(data_zip_file: Path, tmp_path: Path, force_download: bool) -> None:
    with (
        patch("arctix.dataset.epic_kitchen_100.download_url_to_file") as download_mock,
        patch(
            "arctix.dataset.epic_kitchen_100.TemporaryDirectory.__enter__",
            Mock(return_value=data_zip_file.parent),
        ),
    ):
        download_data(tmp_path, force_download=force_download)
        download_mock.assert_called_once_with(
            ANNOTATION_URL, data_zip_file.as_posix(), progress=True
        )
        assert all(tmp_path.joinpath(filename).is_file() for filename in ANNOTATION_FILENAMES)


def test_download_data_already_exists_force_download_false(tmp_path: Path) -> None:
    with patch(
        "arctix.dataset.epic_kitchen_100.is_annotation_path_ready",
        Mock(return_value=True),
    ):
        download_data(tmp_path)
        # The file should not exist because the download step is skipped
        assert not tmp_path.joinpath("EPIC_100_train.csv").is_file()


def test_download_data_already_exists_force_download_true(
    data_zip_file: Path, tmp_path: Path
) -> None:
    with (
        patch(
            "arctix.dataset.epic_kitchen_100.is_annotation_path_ready",
            Mock(return_value=True),
        ),
        patch(
            "arctix.dataset.epic_kitchen_100.TemporaryDirectory.__enter__",
            Mock(return_value=data_zip_file.parent),
        ),
        patch("arctix.dataset.epic_kitchen_100.download_url_to_file") as download_mock,
    ):
        download_data(tmp_path, force_download=True)
        download_mock.assert_called_once_with(
            ANNOTATION_URL, data_zip_file.as_posix(), progress=True
        )
        assert all(tmp_path.joinpath(filename).is_file() for filename in ANNOTATION_FILENAMES)


##############################################
#     Tests for is_annotation_path_ready     #
##############################################


def test_is_annotation_path_ready_true(tmp_path: Path) -> None:
    for filename in ANNOTATION_FILENAMES:
        save_text("", tmp_path.joinpath(filename))
    assert is_annotation_path_ready(tmp_path)


def test_is_annotation_path_ready_false_missing_all(tmp_path: Path) -> None:
    assert not is_annotation_path_ready(tmp_path)


def test_is_annotation_path_ready_false_missing_partial(tmp_path: Path) -> None:
    for filename in ANNOTATION_FILENAMES[::2]:
        save_text("", tmp_path.joinpath(filename))
    assert not is_annotation_path_ready(tmp_path)


###############################
#     Tests for load_data     #
###############################


def test_load_data(
    data_dir: Path,
    data_raw: pl.DataFrame,
    noun_vocab: Vocabulary,
    verb_vocab: Vocabulary,
) -> None:
    data, metadata = load_data(data_dir, split="train")
    assert_frame_equal(data, data_raw)
    assert objects_are_equal(metadata, {"noun_vocab": noun_vocab, "verb_vocab": verb_vocab})


#####################################
#     Tests for load_event_data     #
#####################################


def test_load_event_data(data_file: Path, data_raw: pl.DataFrame) -> None:
    assert_frame_equal(load_event_data(data_file), data_raw)


def test_load_data_empty(empty_data_file: Path) -> None:
    assert_frame_equal(
        load_event_data(empty_data_file),
        pl.DataFrame(
            {
                Column.ALL_NOUN_IDS: [],
                Column.ALL_NOUNS: [],
                Column.NARRATION: [],
                Column.NARRATION_ID: [],
                Column.NARRATION_TIMESTAMP: [],
                Column.NOUN: [],
                Column.NOUN_ID: [],
                Column.PARTICIPANT_ID: [],
                Column.START_FRAME: [],
                Column.START_TIMESTAMP: [],
                Column.STOP_FRAME: [],
                Column.STOP_TIMESTAMP: [],
                Column.VERB: [],
                Column.VERB_ID: [],
                Column.VIDEO_ID: [],
            },
            schema={
                Column.ALL_NOUN_IDS: pl.List(pl.Int64),
                Column.ALL_NOUNS: pl.List(pl.String),
                Column.NARRATION: pl.String,
                Column.NARRATION_ID: pl.String,
                Column.NARRATION_TIMESTAMP: pl.Time,
                Column.NOUN: pl.String,
                Column.NOUN_ID: pl.Int64,
                Column.PARTICIPANT_ID: pl.String,
                Column.START_FRAME: pl.Int64,
                Column.START_TIMESTAMP: pl.Time,
                Column.STOP_FRAME: pl.Int64,
                Column.STOP_TIMESTAMP: pl.Time,
                Column.VERB: pl.String,
                Column.VERB_ID: pl.Int64,
                Column.VIDEO_ID: pl.String,
            },
        ),
    )


#####################################
#     Tests for load_noun_vocab     #
#####################################


def test_load_noun_vocab(data_dir: Path, noun_vocab: Vocabulary) -> None:
    assert load_noun_vocab(data_dir).equal(noun_vocab)


def test_load_noun_vocab_incorrect(empty_noun_file: Path) -> None:
    with pytest.raises(RuntimeError, match="Expected 300 nouns but received 0"):
        load_noun_vocab(empty_noun_file.parent)


#####################################
#     Tests for load_verb_vocab     #
#####################################


def test_load_verb_vocab(data_dir: Path, verb_vocab: Vocabulary) -> None:
    assert load_verb_vocab(data_dir).equal(verb_vocab)


def test_load_verb_vocab_incorrect(empty_verb_file: Path) -> None:
    with pytest.raises(RuntimeError, match="Expected 97 verbs but received 0"):
        load_verb_vocab(empty_verb_file.parent)


##################################
#     Tests for prepare_data     #
##################################


def test_prepare_data(
    data_raw: pl.DataFrame,
    data_prepared: pl.DataFrame,
    noun_vocab: Vocabulary,
    verb_vocab: Vocabulary,
) -> None:
    data, metadata = prepare_data(
        data_raw, metadata={"noun_vocab": noun_vocab, "verb_vocab": verb_vocab}
    )
    assert_frame_equal(data, data_prepared)
    assert objects_are_equal(metadata, {"noun_vocab": noun_vocab, "verb_vocab": verb_vocab})


def test_prepare_data_empty() -> None:
    data, metadata = prepare_data(
        frame=pl.DataFrame(
            {
                Column.ALL_NOUN_IDS: [],
                Column.ALL_NOUNS: [],
                Column.NARRATION: [],
                Column.NARRATION_ID: [],
                Column.NARRATION_TIMESTAMP: [],
                Column.NOUN: [],
                Column.NOUN_ID: [],
                Column.PARTICIPANT_ID: [],
                Column.START_FRAME: [],
                Column.START_TIMESTAMP: [],
                Column.STOP_FRAME: [],
                Column.STOP_TIMESTAMP: [],
                Column.VERB: [],
                Column.VERB_ID: [],
                Column.VIDEO_ID: [],
            },
            schema={
                Column.ALL_NOUN_IDS: pl.List(pl.Int64),
                Column.ALL_NOUNS: pl.List(pl.String),
                Column.NARRATION: pl.String,
                Column.NARRATION_ID: pl.String,
                Column.NARRATION_TIMESTAMP: pl.Time,
                Column.NOUN: pl.String,
                Column.NOUN_ID: pl.Int64,
                Column.PARTICIPANT_ID: pl.String,
                Column.START_FRAME: pl.Int64,
                Column.START_TIMESTAMP: pl.Time,
                Column.STOP_FRAME: pl.Int64,
                Column.STOP_TIMESTAMP: pl.Time,
                Column.VERB: pl.String,
                Column.VERB_ID: pl.Int64,
                Column.VIDEO_ID: pl.String,
            },
        ),
        metadata={},
    )
    assert_frame_equal(
        data,
        pl.DataFrame(
            {
                Column.ALL_NOUN_IDS: [],
                Column.ALL_NOUNS: [],
                Column.NARRATION: [],
                Column.NARRATION_ID: [],
                Column.NARRATION_TIMESTAMP: [],
                Column.NOUN: [],
                Column.NOUN_ID: [],
                Column.PARTICIPANT_ID: [],
                Column.START_FRAME: [],
                Column.START_TIME_SECOND: [],
                Column.START_TIMESTAMP: [],
                Column.STOP_FRAME: [],
                Column.STOP_TIME_SECOND: [],
                Column.STOP_TIMESTAMP: [],
                Column.VERB: [],
                Column.VERB_ID: [],
                Column.VIDEO_ID: [],
            },
            schema={
                Column.ALL_NOUN_IDS: pl.List(pl.Int64),
                Column.ALL_NOUNS: pl.List(pl.String),
                Column.NARRATION: pl.String,
                Column.NARRATION_ID: pl.String,
                Column.NARRATION_TIMESTAMP: pl.Time,
                Column.NOUN: pl.String,
                Column.NOUN_ID: pl.Int64,
                Column.PARTICIPANT_ID: pl.String,
                Column.START_FRAME: pl.Int64,
                Column.START_TIME_SECOND: pl.Float32,
                Column.START_TIMESTAMP: pl.Time,
                Column.STOP_FRAME: pl.Int64,
                Column.STOP_TIME_SECOND: pl.Float32,
                Column.STOP_TIMESTAMP: pl.Time,
                Column.VERB: pl.String,
                Column.VERB_ID: pl.Int64,
                Column.VIDEO_ID: pl.String,
            },
        ),
    )
    assert objects_are_equal(metadata, {})
