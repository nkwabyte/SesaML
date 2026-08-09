from typing import List
import numpy as np

def levenshtein_distance(ref: List[str], hyp: List[str]) -> int:
    """Calculates Levenshtein distance between reference and hypothesis tokens."""
    r_len, h_len = len(ref), len(hyp)
    dp = np.zeros((r_len + 1, h_len + 1), dtype=int)

    for i in range(r_len + 1):
        dp[i][0] = i
    for j in range(h_len + 1):
        dp[0][j] = j

    for i in range(1, r_len + 1):
        for j in range(1, h_len + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    return dp[r_len][h_len]

def calculate_wer(reference: str, hypothesis: str) -> float:
    """Calculates Word Error Rate (WER) between reference and hypothesis strings."""
    try:
        import jiwer
        return float(jiwer.wer(reference, hypothesis))
    except ImportError:
        ref_words = reference.strip().split()
        hyp_words = hypothesis.strip().split()
        if len(ref_words) == 0:
            return 0.0 if len(hyp_words) == 0 else 1.0
        dist = levenshtein_distance(ref_words, hyp_words)
        return float(dist / len(ref_words))

def calculate_cer(reference: str, hypothesis: str) -> float:
    """Calculates Character Error Rate (CER) between reference and hypothesis strings."""
    try:
        import jiwer
        return float(jiwer.cer(reference, hypothesis))
    except ImportError:
        ref_chars = list(reference.strip())
        hyp_chars = list(hypothesis.strip())
        if len(ref_chars) == 0:
            return 0.0 if len(hyp_chars) == 0 else 1.0
        dist = levenshtein_distance(ref_chars, hyp_chars)
        return float(dist / len(ref_chars))
