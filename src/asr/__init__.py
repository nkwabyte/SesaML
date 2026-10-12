"""
Automatic speech recognition: acoustic models, the audio data pipeline, CTC
decoding, training, inference and speaker diarization.

Kept apart from the shared layer (`src.config`, `src.utils`) so that the
translation model added in phase two is a sibling of this package rather than
tangled through it.
"""
