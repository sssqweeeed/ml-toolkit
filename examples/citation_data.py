"""Real observed-link ranking data from the original LINQS Cora archive.

Recorded citations are positive observations. Other directed paper pairs are
unlabeled candidates: a missing record is not evidence of a confirmed negative.
This is an open-world retrieval illustration, not an author-provided PU
benchmark. Scores are cosine similarities between the original binary word
vectors, not calibrated citation probabilities or a fitted PU classifier.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import logging
from pathlib import Path
from tarfile import TarInfo, open as open_tar
from tempfile import TemporaryDirectory
from urllib.error import URLError
from urllib.request import urlopen

import numpy as np

from ml_toolkit.types import FloatArray, IntArray

CORA_URL = "https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz"
CORA_MIRROR = (
    "https://raw.githubusercontent.com/tkipf/pygcn/"
    "1600b5b748b3976413d1e307540ccc62605b4d6d/data/cora"
)
CORA_FILE_HASHES = {
    "cora.content": "b8ebed0fd45e53995abafd1b70cf6a9fcc8f86abc5826d69b97fe6ca46e9c8ee",
    "cora.cites": "ec1a372391b7f0f60a6aff0084e8abd8f19f0faa7e1f2441a41c492042d5945e",
}
DEFAULT_CITATION_CACHE = Path(__file__).resolve().parents[1] / ".cache/linqs-cora"
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CitationGallery:
    """Store recorded citations and unverified candidates with content scores.

    Attributes
    ----------
    labels : FloatArray
        One for recorded citations and zero for unlabeled candidate pairs.
    scores : FloatArray
        Aligned cosine word-vector similarities in [0, 1].
    pairs : tuple[tuple[str, str], ...]
        Directed (citing paper ID, cited paper ID) pairs. Recorded links precede
        candidates; no pair is repeated and no paper is paired with itself.
    known_citation_count, unlabeled_pair_count : int
        Numbers of preserved source links and sampled candidate pairs.
    paper_count, feature_count : int
        Number of real source papers and binary word indicators per paper.
    """

    labels: FloatArray
    scores: FloatArray
    pairs: tuple[tuple[str, str], ...]
    known_citation_count: int
    unlabeled_pair_count: int
    paper_count: int
    feature_count: int


def citation_cache_available(cache_dir: Path | None = None) -> bool:
    """Check for the source archive without downloading or writing files.

    Parameters
    ----------
    cache_dir : pathlib.Path or None, default None
        Cache directory; None uses .cache/linqs-cora in the repository.

    Returns
    -------
    bool
        Whether the selected directory contains cora.tgz.
    """
    directory = DEFAULT_CITATION_CACHE if cache_dir is None else cache_dir
    return (directory / "cora.tgz").is_file()


def _download_source(path: Path) -> None:
    try:
        with urlopen(CORA_URL, timeout=30) as response:
            original = response.read()
    except (URLError, TimeoutError) as error:
        logger.warning("Cora source unavailable (%s); using pinned pygcn mirror", error)
        with open_tar(path, "w:gz") as archive:
            for name, expected in CORA_FILE_HASHES.items():
                with urlopen(f"{CORA_MIRROR}/{name}", timeout=30) as response:
                    content = response.read()
                if sha256(content).hexdigest() != expected:
                    raise ValueError(f"Cora mirror checksum mismatch for {name}")
                member = TarInfo(f"cora/{name}")
                member.size = len(content)
                archive.addfile(member, BytesIO(content))
    else:
        path.write_bytes(original)


def _source_records(
    cache_dir: Path, download_if_missing: bool
) -> tuple[tuple[str, ...], FloatArray, IntArray]:
    path = cache_dir / "cora.tgz"
    if not path.is_file():
        if not download_if_missing:
            raise FileNotFoundError(f"Cora data is not cached at {path}")
        cache_dir.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix="cora-download-", dir=cache_dir) as temporary:
            downloaded = Path(temporary) / "cora.tgz"
            _download_source(downloaded)
            downloaded.replace(path)
    with open_tar(path, "r:gz") as archive:
        content = archive.extractfile("cora/cora.content")
        citations = archive.extractfile("cora/cora.cites")
        if content is None or citations is None:
            raise ValueError("The Cora archive must contain its content and citation files")
        rows = [line.split() for line in content.read().decode("utf-8").splitlines()]
        raw_links = [line.split() for line in citations.read().decode("utf-8").splitlines()]
    paper_ids = tuple(row[0] for row in rows)
    if len(set(paper_ids)) != len(paper_ids):
        raise ValueError("Cora paper IDs must be unique")
    features = np.asarray([row[1:-1] for row in rows], dtype=np.float64)
    if features.ndim != 2 or not len(features) or features.shape[1] == 0:
        raise ValueError("Cora word vectors must be a nonempty matrix")
    if not np.isin(features, [0.0, 1.0]).all() or (features.sum(axis=1) == 0).any():
        raise ValueError("Cora word vectors must be binary and have positive norm")
    positions = {paper_id: index for index, paper_id in enumerate(paper_ids)}
    # The source README stores each citation as (cited ID, citing ID).
    # Retain source order and remove exact duplicate directed records only.
    links = tuple(dict.fromkeys((positions[citing], positions[cited]) for cited, citing in raw_links))
    if any(citing == cited for citing, cited in links):
        raise ValueError("Cora recorded citations must not contain self-links")
    return paper_ids, features, np.asarray(links, dtype=np.int64)


def _candidate_pairs(
    observed: IntArray, paper_count: int, count: int
) -> IntArray:
    recorded = {tuple(pair) for pair in observed.tolist()}
    available = paper_count * (paper_count - 1) - len(recorded)
    if not 1 <= count <= available:
        raise ValueError("unlabeled_pair_count must fit the unrecorded non-self pairs")
    generator = np.random.default_rng(42)
    candidates: list[tuple[int, int]] = []
    selected: set[tuple[int, int]] = set()
    while len(candidates) < count:
        draws = generator.integers(0, paper_count, size=(2 * (count - len(candidates)), 2))
        for citing, cited in draws.tolist():
            pair = (citing, cited)
            if citing != cited and pair not in recorded and pair not in selected:
                selected.add(pair)
                candidates.append(pair)
                if len(candidates) == count:
                    break
    return np.asarray(candidates, dtype=np.int64)


def load_citation_gallery(
    *,
    cache_dir: Path | None = None,
    download_if_missing: bool = True,
    unlabeled_pair_count: int = 20_000,
) -> CitationGallery:
    """Score real citations and unverified candidate pairs by word similarity.

    Parameters
    ----------
    cache_dir : pathlib.Path or None, default None
        Cache directory for the original archive. None uses .cache/linqs-cora.
    download_if_missing : bool, default True
        Allow downloading the LINQS archive, with a pinned, checksum-verified
        pygcn mirror as fallback. False raises for a missing cache without
        making network requests or writing files.
    unlabeled_pair_count : int, default 20000
        Number of distinct absent directed pairs sampled uniformly with seed 42.
        Must be positive and fit the available non-self, unrecorded pairs.

    Returns
    -------
    CitationGallery
        All 5,429 recorded citations and sampled unverified candidate pairs,
        aligned with cosine similarity scores from 1,433 source word indicators.

    Notes
    -----
    Reads only cora.content and cora.cites; archive paths are never extracted.
    Topic labels are ignored. Candidate sampling never alters or hides observed
    citation labels and never uses the similarity scores. No prior prevalence,
    supervised labels, invented dates, or simulated feature values are used.
    A sampled candidate set does not estimate the prevalence of real citations.
    """
    directory = DEFAULT_CITATION_CACHE if cache_dir is None else cache_dir
    paper_ids, features, observed = _source_records(directory, download_if_missing)
    candidates = _candidate_pairs(observed, len(paper_ids), unlabeled_pair_count)
    pairs = np.concatenate((observed, candidates))
    norms = np.linalg.norm(features, axis=1)
    scores = np.empty(len(pairs), dtype=np.float64)
    # Batches keep the two indexed word-vector matrices bounded in memory.
    for start in range(0, len(pairs), 1024):
        batch = pairs[start:start + 1024]
        left, right = batch[:, 0], batch[:, 1]
        numerator = np.einsum("ij,ij->i", features[left], features[right])
        scores[start:start + len(batch)] = numerator / (norms[left] * norms[right])
    scores = np.clip(scores, 0.0, 1.0)
    return CitationGallery(
        labels=np.concatenate((np.ones(len(observed)), np.zeros(len(candidates)))),
        scores=scores,
        pairs=tuple((paper_ids[int(citing)], paper_ids[int(cited)]) for citing, cited in pairs),
        known_citation_count=len(observed),
        unlabeled_pair_count=len(candidates),
        paper_count=len(paper_ids),
        feature_count=features.shape[1],
    )
