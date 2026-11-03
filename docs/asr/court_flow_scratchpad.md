# Court-flow scratchpad

Status board for [`court_flow.md`](court_flow.md). The plan stays there. This page only records where the work stands.

Update the date and the log when a phase moves. Do not promote a model from this page.

**As of 2026-10-31.** Implementation has not started. Served model is `conformer-medium/v003`.

## Now

| | |
|---|---|
| Current phase | P0 recorded. Next code is P1, then P2. Neither is started. |
| Blocker | None for P1. P3 waits on a re-draft (P2) and then a speaker. |
| Served pointer | `conformer-medium` current is `v003` |
| Court sheet | `data/court_gold/pack/transcription_sheet.csv`, 120 rows, `corrected` filled on 0 |
| Drafts on that sheet | The existing Conformer drafts. They are not pinned to v003. |
| Private key | `data/court_gold/private/key.csv` is on this Ubuntu machine. It does not go to `knust-gpu` or the MacBook. |

## Phases

- [x] **P0 — Freeze v003.** Decision written in `court_flow.md`. Pointer checked: still `v003`.
- [ ] **P1 — Training must not promote.** Not started. `Trainer._publish` still promotes when dev WER beats the stored test WER.
- [ ] **P2 — Draft the pack with v003.** Not started. Do this before anyone fills `corrected`. The column is empty, so a rebuild will not overwrite speaker work.
- [ ] **P3 — Speaker fills the sheet.** Not started. Blocked on P2.
- [ ] **P4 — Score the pilot.** Not started. Blocked on P3. From-scratch arm is 20 clips. Training comparison waits for about 100.
- [ ] **P5 — Drop Fante-marked WAXAL.** Not started. Blocked on the P4 gate.
- [ ] **P6 — Court language model.** Not started. Blocked on having court text from P3. Separate from P5.
- [ ] **P7 — More court hours.** Not started. Blocked on scored from-scratch references and a checked sample.

## Log

- **2026-10-31.** Plan and this scratchpad written. No code, no re-draft, no training. Sheet has 120 rows and no `corrected` text. Registry current for `conformer-medium` is `v003`.
- **2026-11-02.** Sheet still empty (120 rows, 0 corrected). No court-flow code started. A separate note, [`ghanaopenai_twi_multispeaker_tts.md`](ghanaopenai_twi_multispeaker_tts.md), records a 91-hour broadcast set to add later. It is not part of P0–P7 and it has not been downloaded.
