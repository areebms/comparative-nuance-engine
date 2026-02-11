from __future__ import annotations

import csv
import io
import json
import os
import tempfile

import numpy as np
from gensim.models import KeyedVectors
from numpy.linalg import norm
from scipy.linalg import orthogonal_procrustes
from sklearn.metrics.pairwise import cosine_similarity

from shared.aws import (
    PipelineTable,
    get_keys_with_prefix,
    get_session,
    load_text_from_s3,
    upload_object,
)
from shared.commons import get_index

VECTOR_SIZE = 200
S3_PREFIX = f"keyed_vector_group_data/"
S3_BUCKET = os.getenv("S3_BUCKET")


def _load_raw_vectors_from_s3(session, s3_prefix):
    model_keys = sorted(
        key for key in get_keys_with_prefix(session, s3_prefix)
        if key.endswith(".model")
    )
    raw_vectors_stack = []
    s3 = session.client("s3")

    for key in model_keys:
        fd, tmp_path = tempfile.mkstemp(suffix=".model")
        os.close(fd)
        try:
            s3.download_file(S3_BUCKET, key, tmp_path)  # streaming download
            raw_vectors_stack.append(KeyedVectors.load(tmp_path))
        finally:
            try:
                os.unlink(tmp_path)
            except FileNotFoundError:
                pass

    return raw_vectors_stack


class VectorsStackAggregator:
    """Aggregate Word2Vec keyed vectors trained on the same text."""

    def __init__(self, session, vectors_stack, index):
        self.vectors_stack = vectors_stack
        self.terms = list(vectors_stack[0].key_to_index)
        self.anchors = self.terms
        self.s3_prefix = f"{S3_PREFIX}{index}/"
        self.session = session

    def align_models(self):
        self.vectors_stack[0].fill_norms(force=True)
        idx_ref = np.array([self.vectors_stack[0].key_to_index[w] for w in self.terms])

        for i in range(1, len(self.vectors_stack)):
            idx_kv = np.array(
                [self.vectors_stack[i].key_to_index[w] for w in self.terms]
            )
            rotation, _ = orthogonal_procrustes(
                self.vectors_stack[i].vectors[idx_kv],
                self.vectors_stack[0].vectors[idx_ref],
            )
            self.vectors_stack[i].vectors = self.vectors_stack[i].vectors @ rotation

            if hasattr(self.vectors_stack[i], "vectors_norm"):
                self.vectors_stack[i].vectors_norm = None
            if hasattr(self.vectors_stack[i], "norms"):
                self.vectors_stack[i].norms = None

            self.vectors_stack[i].fill_norms(force=True)

    def collect_vectors_by_term(self):
        self.vectors_by_term = {}
        for raw_vectors in self.vectors_stack:
            for term in self.terms:
                self.vectors_by_term.setdefault(term, []).append(raw_vectors[term])

    def get_centroid_model(self):
        self.centroid_keyed_vector = KeyedVectors(vector_size=VECTOR_SIZE)
        if not hasattr(self, "vectors_by_term"):
            self.collect_vectors_by_term()
        self.centroid_keyed_vector.add_vectors(
            self.terms,
            np.stack(
                [np.mean(self.vectors_by_term[term], axis=0) for term in self.terms]
            ).astype(np.float32),
        )
        self.centroid_keyed_vector.fill_norms(force=True)
        return self.centroid_keyed_vector

    def get_term_stability_data(self):
        if not hasattr(self, "vectors_by_term"):
            self.collect_vectors_by_term()
        self.term_stability_data = []
        for term, vectors in self.vectors_by_term.items():
            vectors = np.stack(vectors)
            centroid = vectors.mean(axis=0)
            lengths = norm(vectors, axis=1)
            vectors_normed = vectors / norm(vectors, axis=1, keepdims=True)
            centroid_normed = (centroid / norm(centroid)).reshape(1, -1)
            self.term_stability_data.append(
                {
                    "term": term,
                    "count": self.vectors_stack[0].get_vecattr(term, "count"),
                    "overall": float(norm(vectors - centroid, axis=1).mean()),
                    "semantic": float(
                        1
                        - cosine_similarity(vectors_normed, centroid_normed)
                        .ravel()
                        .mean()
                    ),
                    "norm": float(np.abs(lengths - lengths.mean()).mean()),
                }
            )
        return self.term_stability_data

    def upload_object(
        self, s3_key, file_bytes, content_type="application/octet-stream"
    ):
        upload_object(
            self.session, f"{self.s3_prefix}{s3_key}", file_bytes, content_type
        )

    def upload_aligned_models(self):
        s3 = self.session.client("s3")  # or however you access it

        with tempfile.TemporaryDirectory() as tmpdir:
            aligned_models_dir = os.path.join(tmpdir, "aligned_models")
            os.makedirs(aligned_models_dir, exist_ok=True)

            for i, raw_vector in enumerate(self.vectors_stack):
                local_model_path = os.path.join(aligned_models_dir, f"{i}.model")
                raw_vector.save(local_model_path)

                s3.upload_file(
                    local_model_path,
                    S3_BUCKET,
                    f"{self.s3_prefix}/aligned_models/{i}.model",
                    ExtraArgs={"ContentType": "application/octet-stream"},
                )

    def upload_centroid_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            centroid_path = os.path.join(tmpdir, "centroid.model")
            self.centroid_keyed_vector.save(centroid_path)
            self.upload_object(
                "centroid.model",
                open(centroid_path, "rb").read(),
                "application/octet-stream",
            )

    def upload_term_stability_data(self):
        headers = list(self.term_stability_data[0].keys())
        buffer = io.StringIO(newline="")
        writer = csv.writer(buffer)
        writer.writerow(headers)
        for row in self.term_stability_data:
            writer.writerow([row[h] for h in headers])
        self.upload_object(
            f"term_stability.csv",
            buffer.getvalue().encode("utf-8"),
            "text/csv; charset=utf-8",
        )

def aggregate_data(index):
    session = get_session()

    table = PipelineTable(session)

    item = table.get(
        index, expression="platform_data,s3_word_vectors_prefix,s3_metadata_key"
    )

    s3_metadata_key = item.get("s3_metadata_key")
    if not s3_metadata_key:
        print(f"{index} has not been scraped.")
        exit()

    metadata = json.loads(load_text_from_s3(session, s3_metadata_key))

    table.update_entry(index, "author", ";".join(metadata["author"]))
    table.update_entry(index, "title", metadata["title"][0])

    s3_word_vectors_prefix = item.get("s3_word_vectors_prefix") if item else None
    if not s3_word_vectors_prefix:
        print(f"{index} does not have enough models.")
        exit()

    vectors_stack = _load_raw_vectors_from_s3(session, s3_word_vectors_prefix)
    if not vectors_stack:
        print(f"No models found for {s3_word_vectors_prefix}.")
        exit()

    aggregator = VectorsStackAggregator(session, vectors_stack, index)
    print("Aligning models")
    aggregator.align_models()
    print("Finding Centroid")
    aggregator.get_centroid_model()
    print("get_term_stability_data")
    aggregator.get_term_stability_data()
    print("upload_aligned_models")
    aggregator.upload_aligned_models()
    print("upload_centroid_data")
    aggregator.upload_centroid_data()
    print("upload_term_stability_data")
    aggregator.upload_term_stability_data()
    print("table.update_entry")
    table.update_entry(index, "s3_aligned_data_prefix", aggregator.s3_prefix)

    print(f"Aggregated {len(vectors_stack)} models for {index}.")


if __name__ == "__main__":
    aggregate_data(get_index())


