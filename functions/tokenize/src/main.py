import csv
import io
import logging

import spacy
from nltk.tokenize import sent_tokenize

from shared.aws import PipelineTable, upload_object, load_text_from_s3, get_session
from shared.commons import get_index

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


SENTENCE_ENDINGS = {".", "!", "?"}


def chunk_text(nlp, text):
    all_sentences = sent_tokenize(text)
    num_chunks = len(all_sentences) % nlp.max_length
    if num_chunks <= 1:
        return [text]

    sentences_per_chunk = int(len(all_sentences) / num_chunks)
    sentence_chunks = []
    current_chunk = []

    for sentence in all_sentences:
        current_chunk.append(sentence)
        if sentence[-1] not in SENTENCE_ENDINGS:
            continue

        if len(current_chunk) >= sentences_per_chunk:
            sentence_chunks.append(current_chunk)
            current_chunk = []

    if current_chunk:
        sentence_chunks.append(current_chunk)

    return [" ".join(chunk) for chunk in sentence_chunks]


def csv_bytes(rows):
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def tokenize(index):
    session = get_session()

    table = PipelineTable(session)

    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "senter"])
    nlp.add_pipe("sentencizer")

    item = table.get(
        index,
        expression="s3_text_key,s3_token_texts_key,s3_token_lemmas_key,s3_token_tags_key",
    )

    s3_text_key = item.get("s3_text_key")
    if not s3_text_key:
        logger.info("Index has not been scraped", extra={"index": index})
        return

    if (
        item.get("s3_token_texts_key")
        and item.get("s3_token_lemmas_key")
        and item.get("s3_token_tags_key")
    ):
        logger.info("Index has already been tokenized", extra={"index": index})
        return

    text = load_text_from_s3(session, s3_text_key)
    doc_texts = chunk_text(nlp, text)

    token_texts = []
    token_lemmas = []
    token_tags = []

    for text in doc_texts:
        doc = nlp(text)
        for sentence in doc.sents:
            token_lemmas.append([])
            token_texts.append([])
            token_tags.append([])
            for token in sentence:
                token_lemmas[-1].append(token.lemma_.lower())
                token_texts[-1].append(token.text)
                token_tags[-1].append(token.tag_)

    s3_writes = [
        ("s3_token_texts_key", f"token_texts/{index}.csv", token_texts),
        ("s3_token_lemmas_key", f"token_lemmas/{index}.csv", token_lemmas),
        ("s3_token_tags_key", f"token_tags/{index}.csv", token_tags),
    ]
    for field, s3_key, rows in s3_writes:
        upload_object(
            session,
            s3_key,
            csv_bytes(rows),
            "text/csv; charset=utf-8",
        )
        table.update_entry(index, field, s3_key)


if __name__ == "__main__":
    tokenize(get_index())
