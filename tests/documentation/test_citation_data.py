"""The retrieval gallery preserves real citations and scores original content."""

from io import BytesIO
from tarfile import TarInfo, open as open_tar

import numpy as np
import pytest

pytestmark = pytest.mark.documentation

from examples.citation_data import (
    DEFAULT_CITATION_CACHE,
    citation_cache_available,
    load_citation_gallery,
)


@pytest.fixture(scope="module")
def source():
    if not citation_cache_available():
        pytest.skip("Cache the original LINQS Cora archive before source-data checks")
    with open_tar(DEFAULT_CITATION_CACHE / "cora.tgz", "r:gz") as archive:
        return {
            name: archive.extractfile(f"cora/{name}").read()
            for name in ("cora.content", "cora.cites")
        }


@pytest.fixture(scope="module")
def gallery(source):
    return load_citation_gallery(download_if_missing=False)


def test_missing_cache_never_downloads_or_writes_when_disabled(tmp_path, monkeypatch):
    def unexpected_download(*args, **kwargs):
        pytest.fail("Source-data tests must never implicitly download")

    monkeypatch.setattr("examples.citation_data.urlretrieve", unexpected_download)
    assert not citation_cache_available(tmp_path)
    with pytest.raises(FileNotFoundError, match="Cora data is not cached"):
        load_citation_gallery(cache_dir=tmp_path, download_if_missing=False)
    assert list(tmp_path.iterdir()) == []


def test_all_recorded_citations_preserve_the_original_directed_status(source, gallery):
    original_links = [line.split() for line in source["cora.cites"].decode().splitlines()]
    expected = tuple(dict.fromkeys((citing, cited) for cited, citing in original_links))
    assert gallery.paper_count == 2708
    assert gallery.feature_count == 1433
    assert gallery.known_citation_count == len(original_links) == len(expected) == 5429
    assert gallery.pairs[:5429] == expected
    np.testing.assert_array_equal(gallery.labels[:5429], np.ones(5429))
    np.testing.assert_array_equal(gallery.labels[5429:], np.zeros(20_000))


def test_candidates_are_distinct_unrecorded_nonself_source_pairs(source, gallery):
    paper_ids = {line.split()[0] for line in source["cora.content"].decode().splitlines()}
    observed = set(gallery.pairs[:gallery.known_citation_count])
    candidates = gallery.pairs[gallery.known_citation_count:]
    assert gallery.unlabeled_pair_count == len(candidates) == 20_000
    assert len(set(gallery.pairs)) == len(gallery.pairs) == 25_429
    assert observed.isdisjoint(candidates)
    assert all(left != right and left in paper_ids and right in paper_ids
               for left, right in gallery.pairs)


def test_scores_are_original_binary_word_cosines_not_assigned_probabilities(source, gallery):
    words = {
        fields[0]: np.asarray(fields[1:-1], dtype=np.float64)
        for fields in (line.split() for line in source["cora.content"].decode().splitlines())
    }
    assert gallery.labels.shape == gallery.scores.shape == (25_429,)
    assert np.isfinite(gallery.scores).all()
    assert ((gallery.scores >= 0) & (gallery.scores <= 1)).all()
    positions = np.linspace(0, len(gallery.pairs) - 1, 100, dtype=np.int64)
    expected = []
    for position in positions:
        citing, cited = gallery.pairs[position]
        left, right = words[citing], words[cited]
        expected.append(float(np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right))))
    np.testing.assert_allclose(gallery.scores[positions], expected)
    assert np.unique(gallery.scores).size > 100


def test_topic_labels_are_unused_archive_paths_are_not_extracted_and_sampling_is_stable(
    source, gallery, tmp_path
):
    rows = [line.split() for line in source["cora.content"].decode().splitlines()]
    changed_content = "\n".join("\t".join([*row[:-1], "unused-topic"]) for row in rows).encode()
    archive_path = tmp_path / "cora.tgz"
    with open_tar(archive_path, "w:gz") as archive:
        for name, data in (
            ("cora/cora.content", changed_content),
            ("cora/cora.cites", source["cora.cites"]),
            ("../must-not-be-extracted.txt", b"ignore this unrelated member"),
        ):
            entry = TarInfo(name)
            entry.size = len(data)
            archive.addfile(entry, BytesIO(data))
    repeated = load_citation_gallery(cache_dir=tmp_path, download_if_missing=False)
    assert repeated.pairs == gallery.pairs
    np.testing.assert_array_equal(repeated.labels, gallery.labels)
    np.testing.assert_array_equal(repeated.scores, gallery.scores)
    assert list(tmp_path.iterdir()) == [archive_path]
    assert not (tmp_path.parent / "must-not-be-extracted.txt").exists()


def test_impossible_candidate_count_is_rejected_without_synthetic_pairs(source):
    with pytest.raises(ValueError, match="unlabeled_pair_count"):
        load_citation_gallery(download_if_missing=False, unlabeled_pair_count=0)
    with pytest.raises(ValueError, match="unlabeled_pair_count"):
        load_citation_gallery(download_if_missing=False, unlabeled_pair_count=2708 * 2707)
