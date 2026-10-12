"""
Tests for the character n-gram language model and CTC beam search.

These exist because greedy decoding takes the argmax at every frame
independently, so nothing prefers a spelling that exists in Twi over one that
does not - which is exactly the model's error profile. Measured on the
validation split, fusing an n-gram model into a prefix beam search took WER from
0.4725 to 0.3936.
"""

import math

import pytest
import torch

from src.config import PipelineConfig
from src.asr.data.text_transform import TextTransform
from src.asr.decoding import CTCDecoder, CharNGramLM, beam_search_decode, build_decoder
from src.asr.training.evaluator import greedy_decoder

TWI = [
    "me din de kwame",
    "wo ho te sɛn",
    "ɛte sɛn wo ho yɛ",
    "awurade ne me botan",
    "medaase paa",
    "me kɔ fie",
]


@pytest.fixture
def lm():
    return CharNGramLM(order=4).train(TWI * 20)


@pytest.fixture
def transform():
    return TextTransform()


@pytest.fixture
def index_to_char(transform):
    return {i: (" " if c == "<SPACE>" else c)
            for i, c in transform.index_map.items() if c != "<BLANK>"}


def peaked(sequence, classes, sharpness=12.0):
    """Log-probs concentrated on a chosen class per frame."""
    log_probs = torch.full((len(sequence), classes), -sharpness)
    for t, index in enumerate(sequence):
        log_probs[t, index] = 0.0
    return log_probs.log_softmax(dim=-1)


def spell(text, transform, blank):
    """Frame sequence that decodes to `text`, blank-separated."""
    sequence = []
    for char in text:
        sequence.append(transform.char_map["<SPACE>" if char == " " else char])
        sequence.append(blank)
    return sequence


# --- language model -------------------------------------------------------

def test_lm_prefers_text_it_was_trained_on(lm):
    assert lm.score("me din de kwame") > lm.score("xq zj vb kwqme")


def test_lm_scores_a_plausible_continuation_higher(lm):
    """After "me di", "n" (as in "me din") should beat an unseen letter."""
    assert lm.log_prob("n", "me di") > lm.log_prob("q", "me di")


def test_lm_backs_off_to_shorter_contexts(lm):
    """An unseen long context must still score, via a shorter one."""
    score = lm.log_prob("e", "zzzzzz")
    assert score > -20.0, "backoff should find 'e' in a shorter context"
    assert math.isfinite(score)


def test_lm_never_returns_negative_infinity(lm):
    """One unknown symbol must not veto an entire beam."""
    assert math.isfinite(lm.log_prob("ñ", "me di"))


def test_lm_pruning_shrinks_the_model():
    """Singletons are mostly typos; pruning them is most of the size win."""
    # The common lines repeat; the rare one appears once and should be pruned.
    model = CharNGramLM(order=4).train(TWI * 20 + ["zqxj vbkw"])
    before = len(model.counts)

    model.prune(min_count=2)

    assert len(model.counts) < before
    # And the rare material is what went: its score should now come from backoff.
    assert model.log_prob("q", "z") < model.log_prob("e", "d")


def test_lm_pruning_keeps_unigrams(lm):
    """The empty context is the final backoff and must stay complete."""
    lm.prune(min_count=1000)
    assert "" in lm.counts


def test_lm_round_trips_through_disk(lm, tmp_path):
    path = str(tmp_path / "lm.json")
    lm.save(path)
    reloaded = CharNGramLM.load(path)

    assert reloaded.order == lm.order
    assert reloaded.score("me din de kwame") == pytest.approx(lm.score("me din de kwame"))


def test_lm_perplexity_is_lower_on_familiar_text(lm):
    assert lm.perplexity(TWI) < lm.perplexity(["xqz jvb wqm", "zzz qqq vvv"])


# --- beam search ----------------------------------------------------------

def test_beam_search_matches_greedy_without_a_language_model(transform, index_to_char):
    """With no LM, beam search must not disagree with greedy on clear audio."""
    blank = transform.blank_label
    log_probs = peaked(spell("medaase", transform, blank), transform.vocab_size)

    greedy = transform.int_to_text(greedy_decoder(log_probs.unsqueeze(1), blank_label=blank)[0])
    beam = beam_search_decode(log_probs, index_to_char, blank,
                              language_model=None, beam_width=10, alpha=0.0, beta=0.0)
    assert beam == greedy == "medaase"


def test_beam_search_preserves_repeated_characters(transform, index_to_char):
    """CTC collapses repeats, so "aa" survives only via the blank between."""
    blank = transform.blank_label
    log_probs = peaked(spell("aa bb", transform, blank), transform.vocab_size)
    assert beam_search_decode(log_probs, index_to_char, blank,
                              beam_width=10, alpha=0.0, beta=0.0) == "aa bb"


def test_beam_search_on_empty_input(transform, index_to_char):
    assert beam_search_decode(torch.zeros(0, transform.vocab_size),
                              index_to_char, transform.blank_label) == ""


def test_beam_search_decodes_all_blanks_to_nothing(transform, index_to_char):
    blank = transform.blank_label
    log_probs = peaked([blank] * 10, transform.vocab_size)
    assert beam_search_decode(log_probs, index_to_char, blank, beam_width=5) == ""


def test_language_model_breaks_an_acoustic_tie(transform, index_to_char, lm):
    """
    Where the acoustics are ambiguous, the LM should pick the real word.

    The frame for the second character is deliberately split between "n" (giving
    "din", which the LM has seen) and "q" (giving "diq", which it has not).
    """
    blank = transform.blank_label
    n, q = transform.char_map["n"], transform.char_map["q"]

    log_probs = torch.full((6, transform.vocab_size), -12.0)
    log_probs[0, transform.char_map["d"]] = 0.0
    log_probs[1, blank] = 0.0
    log_probs[2, transform.char_map["i"]] = 0.0
    log_probs[3, blank] = 0.0
    log_probs[4, n] = 0.0
    log_probs[4, q] = 0.0          # exactly tied on the acoustics
    log_probs[5, blank] = 0.0
    log_probs = log_probs.log_softmax(dim=-1)

    with_lm = beam_search_decode(log_probs, index_to_char, blank, language_model=lm,
                                 beam_width=15, alpha=2.0, beta=0.0)
    assert with_lm == "din", f"the language model should have chosen 'din', got {with_lm!r}"


def test_alpha_zero_ignores_the_language_model(transform, index_to_char, lm):
    """alpha=0 must reproduce plain beam search, so the sweep has a control."""
    blank = transform.blank_label
    log_probs = peaked(spell("medaase", transform, blank), transform.vocab_size)

    without = beam_search_decode(log_probs, index_to_char, blank, language_model=None,
                                 beam_width=10, alpha=0.0, beta=0.0)
    with_zero = beam_search_decode(log_probs, index_to_char, blank, language_model=lm,
                                   beam_width=10, alpha=0.0, beta=0.0)
    assert without == with_zero


# --- the shared decoder ---------------------------------------------------

def test_build_decoder_honours_the_config(transform, tmp_path):
    config = PipelineConfig()
    config.decoding.decoder = "greedy"
    assert build_decoder(config, transform).name == "greedy"


def test_decoder_falls_back_when_the_lm_file_is_absent(transform):
    """A missing LM must degrade to beam search, not refuse to transcribe."""
    decoder = CTCDecoder(transform, decoder="beam", lm_path="does/not/exist.json")
    assert decoder.language_model is None
    assert decoder.name == "beam"


def test_decoder_batch_honours_per_item_lengths(transform):
    """Decoding past an item's real length would transcribe padding."""
    blank = transform.blank_label
    decoder = CTCDecoder(transform, decoder="greedy")

    batch = torch.stack([
        peaked(spell("me", transform, blank) + [blank] * 6, transform.vocab_size),
        peaked(spell("wo", transform, blank) + [blank] * 6, transform.vocab_size),
    ])
    assert decoder.decode_batch(batch, lengths=torch.tensor([4, 4])) == ["me", "wo"]


def test_decoder_describes_itself(transform, tmp_path):
    path = str(tmp_path / "lm.json")
    CharNGramLM(order=3).train(TWI).save(path)

    decoder = CTCDecoder(transform, decoder="beam", lm_path=path, alpha=0.5, beta=0.5)
    described = decoder.describe()

    assert described["decoder"] == "beam+lm"
    assert described["alpha"] == 0.5
    assert described["lm_contexts"] > 0
