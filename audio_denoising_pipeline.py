
#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse
import os
import sys
import math
import json
import hashlib
import warnings
from pathlib import Path
from functools import partial
from multiprocessing import Pool, cpu_count
import tempfile

import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from tqdm import tqdm
from scipy.signal import wiener

warnings.filterwarnings("ignore")

try:
    import noisereduce as nr
    HAVE_NR = True
except Exception:
    HAVE_NR = False

try:
    import pyloudnorm as pyln
    HAVE_LOUDNORM = True
except Exception:
    HAVE_LOUDNORM = False

AUDIO_EXTS = {".wav", ".flac", ".ogg", ".mp3", ".m4a", ".aac"}

def list_audio_files(root: Path):
    return sorted([p for p in root.rglob("*") if p.suffix.lower() in AUDIO_EXTS])

def hash_params(d: dict) -> str:
    js = json.dumps(d, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(js.encode("utf-8")).hexdigest()[:8]

# ---------- Metrics helpers ----------

def frame_signal_safe(y, frame_length=2048, hop_length=512):
    if len(y) == 0:
        return np.empty((0, frame_length), dtype=np.float32)
    if len(y) < frame_length:
        y = np.pad(y, (0, frame_length - len(y)), mode="constant")
    return librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length).T

def estimate_noise_mask(y, frame_length=2048, hop_length=512, percentile=20):
    frames = frame_signal_safe(y, frame_length, hop_length)
    if frames.shape[0] == 0:
        return np.zeros((0,), dtype=bool)
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    thresh = np.percentile(rms, percentile)
    return rms <= thresh

def estimate_noise_power_from_frames(frames, mask):
    if frames.size == 0 or mask.size == 0 or not mask.any():
        return 1e-8, 0
    noise_frames = frames[mask]
    per_frame_power = np.mean(noise_frames ** 2, axis=1)
    return float(np.mean(per_frame_power)), int(noise_frames.shape[0])

def estimated_snr_db_from_totals(total_energy_sum, total_samples, noise_power_est):
    if total_samples <= 0:
        return 0.0
    total_power = float(total_energy_sum / total_samples + 1e-12)
    signal_power_est = max(total_power - noise_power_est, 1e-12)
    return 10.0 * math.log10(signal_power_est / (noise_power_est + 1e-12))

def spectral_distance(y_ref, y_hat, n_fft=1024, hop_length=256):
    S_ref = np.abs(librosa.stft(y_ref, n_fft=n_fft, hop_length=hop_length)) + 1e-9
    S_hat = np.abs(librosa.stft(y_hat, n_fft=n_fft, hop_length=hop_length)) + 1e-9
    L_ref = np.log(S_ref)
    L_hat = np.log(S_hat)
    T = min(L_ref.shape[1], L_hat.shape[1])
    return float(np.mean((L_ref[:, :T] - L_hat[:, :T]) ** 2))

def spectral_distance_chunk(L_ref, L_hat):
    T = min(L_ref.shape[1], L_hat.shape[1])
    if T <= 0:
        return 0.0, 0
    diff = (L_ref[:, :T] - L_hat[:, :T]) ** 2
    return float(np.sum(diff)), int(diff.size)

# ---------- Normalization helpers ----------

def compute_peak_gain_from_peak(peak, target_dbfs=-1.0):
    target_lin = 10 ** (target_dbfs / 20.0)
    if peak <= 0:
        return 1.0
    return min(target_lin / peak, 10.0)  # clamp absurd gains

def compute_rms_gain_from_rms(rms, target_dbfs=-20.0):
    target_lin = 10 ** (target_dbfs / 20.0)
    if rms <= 0:
        return 1.0
    return min(target_lin / rms, 10.0)

def apply_gain_clip(y, gain):
    if gain == 1.0:
        return y
    out = y * gain
    return np.clip(out, -1.0, 1.0).astype(np.float32)

def fullbuffer_normalize(y, sr, normalize, target_dbfs, target_lufs):
    if normalize == "none":
        return y
    if normalize == "peak":
        peak = float(np.max(np.abs(y)) + 1e-12)
        g = compute_peak_gain_from_peak(peak, target_dbfs)
        return apply_gain_clip(y, g)
    if normalize == "rms":
        rms = float(np.sqrt(np.mean(y**2)) + 1e-12)
        g = compute_rms_gain_from_rms(rms, target_dbfs)
        return apply_gain_clip(y, g)
    if normalize == "lufs":
        if not HAVE_LOUDNORM:
            raise RuntimeError("LUFS normalization requires pyloudnorm. Install with: pip install pyloudnorm")
        meter = pyln.Meter(sr)
        loud = meter.integrated_loudness(y.astype(np.float32))
        y_norm = pyln.normalize.loudness(y.astype(np.float32), loud, target_lufs)
        return np.clip(y_norm, -1.0, 1.0).astype(np.float32)
    raise ValueError(f"Unknown normalize: {normalize}")

# ---------- Denoisers ----------

def denoise_spectral_gate(y, sr, nr_kwargs=None):
    if not HAVE_NR:
        raise RuntimeError("noisereduce is not installed. `pip install noisereduce`")
    nr_kwargs = nr_kwargs or {}
    defaults = dict(
        stationary=False,
        prop_decrease=1.0,
        n_fft=1024,
        win_length=1024,
        hop_length=256,
        time_constant_s=0.4,
        freq_mask_smooth_hz=500,
        time_mask_smooth_ms=80,
    )
    for k, v in defaults.items():
        nr_kwargs.setdefault(k, v)
    y_hat = nr.reduce_noise(y=y, sr=sr, **nr_kwargs)
    mx = np.max(np.abs(y_hat)) + 1e-12
    if mx > 1.0:
        y_hat = y_hat / mx
    return y_hat.astype(np.float32), nr_kwargs

def denoise_wiener(y, sr, n_fft=1024, hop_length=256, wiener_sz=3):
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    mag, phase = np.abs(S), np.angle(S)
    mag_w = wiener(mag, mysize=wiener_sz)
    S_hat = mag_w * np.exp(1j * phase)
    y_hat = librosa.istft(S_hat, hop_length=hop_length)
    y_hat = np.ascontiguousarray(y_hat, dtype=np.float32)
    mx = np.max(np.abs(y_hat)) + 1e-12
    if mx > 1.0:
        y_hat = y_hat / mx
    params = {"n_fft": n_fft, "hop_length": hop_length, "wiener_size": wiener_sz}
    return y_hat.astype(np.float32), params

def apply_denoise(y, sr, method, method_params):
    method = method.lower()
    if method == "spectral_gate":
        y_hat, used = denoise_spectral_gate(y, sr, nr_kwargs=method_params or {})
    elif method == "wiener":
        n_fft = method_params.get("n_fft", 1024) if method_params else 1024
        hop = method_params.get("hop_length", 256) if method_params else 256
        wsize = method_params.get("wiener_size", 3) if method_params else 3
        y_hat, used = denoise_wiener(y, sr, n_fft=n_fft, hop_length=hop, wiener_sz=wsize)
    else:
        raise ValueError(f"Unknown method: {method}")
    return y_hat, used

# ---------- Full-buffer processing ----------

def load_audio_full(path: Path, sr: int):
    y, sr_out = librosa.load(str(path), sr=sr, mono=True)
    if np.abs(np.mean(y)) > 1e-4:
        y = y - np.mean(y)
    return y.astype(np.float32), sr_out

def save_audio(path: Path, y: np.ndarray, sr: int, subtype="PCM_16"):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), y, sr, subtype=subtype)

def process_one_full(
    path_in: Path,
    root_in: Path,
    root_out: Path,
    sr: int,
    method: str,
    method_params: dict,
    frame_length: int,
    hop_length: int,
    save_intermediate: bool,
    normalize: str,
    target_dbfs: float,
    target_lufs: float,
    writer_subtype: str,
    param_tag_extra: dict = None,
):
    rel = path_in.relative_to(root_in)
    tag_dict = {"method": method, "params": method_params,
                "normalize": normalize, "target_dbfs": target_dbfs,
                "target_lufs": target_lufs, "writer_subtype": writer_subtype}
    if param_tag_extra:
        tag_dict.update(param_tag_extra)
    param_tag = hash_params(tag_dict)
    out_path = (root_out / rel).with_suffix(f".{param_tag}.wav")

    if out_path.exists():
        return {"file": str(path_in), "output": str(out_path), "skipped": True, "ok": True,
                "msg": "exists", "pre_snr_db": None, "post_snr_db": None,
                "snr_improvement_db": None, "spec_dist": None,
                "method": method, "params": json.dumps(tag_dict)}

    try:
        y, sr_loaded = load_audio_full(path_in, sr)

        # Pre metrics
        frames = frame_signal_safe(y, frame_length, hop_length)
        mask = estimate_noise_mask(y, frame_length, hop_length, percentile=20)
        noise_power_before, n_noise_frames = estimate_noise_power_from_frames(frames, mask)
        pre_snr = estimated_snr_db_from_totals(float(np.sum(y**2)), len(y), noise_power_before)

        # Denoise
        y_hat, used_params = apply_denoise(y, sr, method, method_params or {})

        # Normalize (full buffer)
        y_hat = fullbuffer_normalize(y_hat, sr, normalize, target_dbfs, target_lufs)

        # Post metrics
        frames_after = frame_signal_safe(y_hat, frame_length, hop_length)
        mask_after = estimate_noise_mask(y_hat, frame_length, hop_length, percentile=20)
        noise_power_after, _ = estimate_noise_power_from_frames(frames_after, mask_after)
        post_snr = estimated_snr_db_from_totals(float(np.sum(y_hat**2)), len(y_hat), noise_power_after)
        spec_dist_val = spectral_distance(y, y_hat)

        save_audio(out_path, y_hat, sr, subtype=writer_subtype)

        return {"file": str(path_in), "output": str(out_path), "skipped": False, "ok": True,
                "msg": "ok", "pre_snr_db": pre_snr, "post_snr_db": post_snr,
                "snr_improvement_db": post_snr - pre_snr, "spec_dist": spec_dist_val,
                "method": method, "params": json.dumps(tag_dict)}
    except Exception as e:
        return {"file": str(path_in), "output": "", "skipped": False, "ok": False,
                "msg": f"error: {e}", "pre_snr_db": None, "post_snr_db": None,
                "snr_improvement_db": None, "spec_dist": None, "method": method,
                "params": json.dumps(tag_dict)}

# ---------- Chunked processing with two-pass normalization ----------

def _chunk_iter(snd, in_sr, out_sr, chunk_duration_s):
    frames_in = int(max(1, round(chunk_duration_s * in_sr)))
    while True:
        y_in = snd.read(frames=frames_in, dtype="float32", always_2d=True)
        if y_in.size == 0:
            break
        if snd.channels > 1:
            y_mono = np.mean(y_in, axis=1)
        else:
            y_mono = y_in.reshape(-1)
        if in_sr != out_sr:
            y = librosa.resample(y_mono, orig_sr=in_sr, target_sr=out_sr)
        else:
            y = y_mono
        if np.abs(np.mean(y)) > 1e-4:
            y = y - np.mean(y)
        yield y.astype(np.float32)

def _overlap_add_write(w, y_chunk, overlap_n, tail_prev, fade_in, fade_out):
    if overlap_n <= 0:
        w.write(y_chunk.astype(np.float32))
        return tail_prev
    if fade_in is None or fade_out is None or len(fade_in) != overlap_n:
        fade_in = np.linspace(0.0, 1.0, overlap_n, dtype=np.float32)
        fade_out = 1.0 - fade_in
    if tail_prev is None:
        if len(y_chunk) <= overlap_n:
            return y_chunk.astype(np.float32)
        else:
            w.write(y_chunk[:-overlap_n].astype(np.float32))
            return y_chunk[-overlap_n:].astype(np.float32)
    else:
        head = y_chunk[:overlap_n].astype(np.float32)
        if len(head) < overlap_n or len(tail_prev) < overlap_n:
            L = min(len(head), len(tail_prev))
            cross = tail_prev[:L] * fade_out[:L] + head[:L] * fade_in[:L]
            w.write(cross)
        else:
            cross = tail_prev * fade_out + head * fade_in
            w.write(cross)
        if len(y_chunk) > overlap_n * 2:
            middle = y_chunk[overlap_n:-overlap_n].astype(np.float32)
            if len(middle) > 0:
                w.write(middle)
        if len(y_chunk) >= overlap_n:
            return y_chunk[-overlap_n:].astype(np.float32)
        else:
            return y_chunk.astype(np.float32)

def process_one_chunked(
    path_in: Path,
    root_in: Path,
    root_out: Path,
    sr: int,
    method: str,
    method_params: dict,
    frame_length: int,
    hop_length: int,
    save_intermediate: bool,
    chunk_duration_s: float,
    chunk_overlap_s: float,
    normalize: str,
    target_dbfs: float,
    target_lufs: float,
    writer_subtype: str = "PCM_16",
):
    rel = path_in.relative_to(root_in)
    tag_dict = {
        "method": method, "params": method_params,
        "chunked": True, "chunk_dur_s": float(chunk_duration_s),
        "chunk_overlap_s": float(chunk_overlap_s),
        "normalize": normalize, "target_dbfs": target_dbfs,
        "target_lufs": target_lufs, "writer_subtype": writer_subtype
    }
    param_tag = hash_params(tag_dict)
    out_path = (root_out / rel).with_suffix(f".{param_tag}.wav")

    if out_path.exists():
        return {"file": str(path_in), "output": str(out_path), "skipped": True, "ok": True,
                "msg": "exists", "pre_snr_db": None, "post_snr_db": None,
                "snr_improvement_db": None, "spec_dist": None,
                "method": method, "params": json.dumps(tag_dict)}

    try:
        # Pass 1: compute metrics & normalization gain (peak/rms) and (optionally) write temp for LUFS
        total_energy_before = total_samples_before = 0
        total_energy_after = total_samples_after = 0
        noise_power_num_before = noise_frames_count_before = 0
        noise_power_num_after = noise_frames_count_after = 0
        specdiff_sum = specdiff_count = 0

        overlap_n = int(max(0, round(chunk_overlap_s * sr)))
        fade_in = fade_out = None
        tail_prev = None

        # For peak/rms we only need to track stats; for LUFS we need a temp denoised file
        temp_path = None
        temp_file = None

        with sf.SoundFile(str(path_in), mode="r") as snd:
            in_sr = snd.samplerate

            # If LUFS, open temp file to store the denoised audio (no normalization yet)
            if normalize == "lufs":
                temp_dir = tempfile.mkdtemp(prefix="denoise_tmp_")
                temp_path = Path(temp_dir) / "tmp_denoised.wav"
                temp_file = sf.SoundFile(str(temp_path), mode="w", samplerate=sr, channels=1, subtype="FLOAT")

            for y in _chunk_iter(snd, in_sr, sr, chunk_duration_s):
                # Pre metrics (chunk)
                frames_b = frame_signal_safe(y, frame_length, hop_length)
                mask_b = estimate_noise_mask(y, frame_length, hop_length, percentile=20)
                noise_power_b, n_noise_b = estimate_noise_power_from_frames(frames_b, mask_b)
                total_energy_before += float(np.sum(y**2))
                total_samples_before += len(y)
                if n_noise_b > 0:
                    per_frame_power = np.mean(frames_b[mask_b] ** 2, axis=1) if frames_b.size else np.array([0.0])
                    noise_power_num_before += float(np.sum(per_frame_power))
                    noise_frames_count_before += int(n_noise_b)

                # Denoise
                y_hat, used_params = apply_denoise(y, sr, method, method_params or {})

                # Post metrics (chunk)
                frames_a = frame_signal_safe(y_hat, frame_length, hop_length)
                mask_a = estimate_noise_mask(y_hat, frame_length, hop_length, percentile=20)
                noise_power_a, n_noise_a = estimate_noise_power_from_frames(frames_a, mask_a)
                total_energy_after += float(np.sum(y_hat**2))
                total_samples_after += len(y_hat)
                if n_noise_a > 0:
                    per_frame_power_a = np.mean(frames_a[mask_a] ** 2, axis=1) if frames_a.size else np.array([0.0])
                    noise_power_num_after += float(np.sum(per_frame_power_a))
                    noise_frames_count_after += int(n_noise_a)

                # Spectral distance acc
                S_ref = np.abs(librosa.stft(y, n_fft=1024, hop_length=256)) + 1e-9
                S_hat = np.abs(librosa.stft(y_hat, n_fft=1024, hop_length=256)) + 1e-9
                L_ref = np.log(S_ref); L_hat = np.log(S_hat)
                ssum, scount = spectral_distance_chunk(L_ref, L_hat)
                specdiff_sum += ssum; specdiff_count += scount

                if normalize == "lufs":
                    # Write raw denoised chunk to temp (no overlap-add during temp; use simple concat)
                    if temp_file is not None:
                        temp_file.write(y_hat.astype(np.float32))

            if temp_file is not None:
                temp_file.close()

        # Compute normalization gain(s)
        noise_power_before_final = (
            noise_power_num_before / max(1, noise_frames_count_before)
            if noise_frames_count_before > 0 else 1e-8
        )
        noise_power_after_final = (
            noise_power_num_after / max(1, noise_frames_count_after)
            if noise_frames_count_after > 0 else 1e-8
        )
        pre_snr = estimated_snr_db_from_totals(total_energy_before, total_samples_before, noise_power_before_final)
        post_snr_unnorm = estimated_snr_db_from_totals(total_energy_after, total_samples_after, noise_power_after_final)
        spec_dist_val = (specdiff_sum / specdiff_count) if specdiff_count > 0 else None

        # Determine global gain for pass 2
        gain = 1.0
        if normalize == "peak":
            global_peak = 0.0
            with sf.SoundFile(str(path_in), mode="r") as snd:
                for y in _chunk_iter(snd, snd.samplerate, sr, chunk_duration_s):
                    y_hat, _ = apply_denoise(y, sr, method, method_params or {})
                    p = float(np.max(np.abs(y_hat)) + 1e-12)
                    if p > global_peak:
                        global_peak = p
            gain = compute_peak_gain_from_peak(global_peak, target_dbfs)
        elif normalize == "rms":
            rms = math.sqrt(total_energy_after / max(1, total_samples_after))
            gain = compute_rms_gain_from_rms(rms, target_dbfs)
        elif normalize == "lufs":
            if not HAVE_LOUDNORM:
                raise RuntimeError("LUFS normalization chosen but pyloudnorm not installed. pip install pyloudnorm")
            y_tmp, _sr_tmp = librosa.load(str(temp_path), sr=sr, mono=True)
            meter = pyln.Meter(sr)
            loud = meter.integrated_loudness(y_tmp.astype(np.float32))
            y_lufs_norm = pyln.normalize.loudness(y_tmp.astype(np.float32), loud, target_lufs)
            peak_before = float(np.max(np.abs(y_tmp)) + 1e-12)
            peak_after = float(np.max(np.abs(y_lufs_norm)) + 1e-12)
            gain = peak_after / peak_before if peak_before > 0 else 1.0
            del y_tmp, y_lufs_norm

        # Pass 2: write final output with overlap-add and global gain
        out_path.parent.mkdir(parents=True, exist_ok=True)
        overlap_n = int(max(0, round(chunk_overlap_s * sr)))
        fade_in = fade_out = None
        tail_prev = None

        with sf.SoundFile(str(path_in), mode="r") as snd, \
             sf.SoundFile(str(out_path), mode="w", samplerate=sr, channels=1, subtype=writer_subtype) as w:
            in_sr = snd.samplerate
            for y in _chunk_iter(snd, in_sr, sr, chunk_duration_s):
                y_hat, _ = apply_denoise(y, sr, method, method_params or {})
                y_hat = apply_gain_clip(y_hat, gain)
                tail_prev = _overlap_add_write(w, y_hat, overlap_n, tail_prev, fade_in, fade_out)
            if overlap_n > 0 and tail_prev is not None and len(tail_prev) > 0:
                w.write(tail_prev.astype(np.float32))

        return {"file": str(path_in), "output": str(out_path), "skipped": False, "ok": True,
                "msg": "ok", "pre_snr_db": pre_snr, "post_snr_db": post_snr_unnorm,
                "snr_improvement_db": post_snr_unnorm - pre_snr, "spec_dist": spec_dist_val,
                "method": method, "params": json.dumps(tag_dict)}

    except Exception as e:
        return {"file": str(path_in), "output": "", "skipped": False, "ok": False,
                "msg": f"error: {e}", "pre_snr_db": None, "post_snr_db": None,
                "snr_improvement_db": None, "spec_dist": None, "method": method,
                "params": json.dumps(tag_dict)}

# ---------- Pilot & Full batch ----------

def run_pilot(files, args, processor_func, param_tag_extra=None):
    print(f"\n▶ Running pilot on {len(files)} file(s) with method={args.method} …")
    records = []
    for p in tqdm(files, desc="Pilot"):
        rec = processor_func(
            path_in=p,
            root_in=args.input_dir,
            root_out=args.output_dir,
            sr=args.sr,
            method=args.method,
            method_params=args.method_params,
            frame_length=args.frame_length,
            hop_length=args.hop_length,
            save_intermediate=args.save_intermediate,
            normalize=args.normalize,
            target_dbfs=args.target_dbfs,
            target_lufs=args.target_lufs,
            writer_subtype=args.writer_subtype,
            **(param_tag_extra or {}),
        )
        records.append(rec)

    df = pd.DataFrame(records)
    pilot_csv = args.output_dir / f"pilot_report_{args.method}.csv"
    df.to_csv(pilot_csv, index=False)
    print(f"✔ Pilot report saved: {pilot_csv}")

    ok = df[df["ok"] & ~df["skipped"]]
    if len(ok):
        print("\nPilot summary (ok & processed):")
        print(
            ok[["snr_improvement_db", "spec_dist"]]
            .describe()
            .round(3)
            .to_string()
        )
        print("\nTips: Higher SNR improvement (dB) is better; lower spectral distance is better.")

    return records

def run_full(files, args, processor_func, param_tag_extra=None):
    print(f"\n▶ Running full batch on {len(files)} file(s) with {args.num_workers} workers …")
    worker = partial(
        processor_func,
        root_in=args.input_dir,
        root_out=args.output_dir,
        sr=args.sr,
        method=args.method,
        method_params=args.method_params,
        frame_length=args.frame_length,
        hop_length=args.hop_length,
        save_intermediate=args.save_intermediate,
        normalize=args.normalize,
        target_dbfs=args.target_dbfs,
        target_lufs=args.target_lufs,
        writer_subtype=args.writer_subtype,
        **(param_tag_extra or {}),
    )

    if args.num_workers <= 1:
        results = [worker(f) for f in tqdm(files, desc="Batch")]
    else:
        with Pool(processes=args.num_workers) as pool:
            results = list(tqdm(pool.imap(worker, files), total=len(files), desc="Batch"))

    df = pd.DataFrame(results)
    out_csv = args.output_dir / f"batch_report_{args.method}.csv"
    df.to_csv(out_csv, index=False)
    print(f"✔ Batch report saved: {out_csv}")

    ok = df[df["ok"] & ~df["skipped"]]
    if len(ok):
        print(
            ok[["snr_improvement_db", "spec_dist"]]
            .describe()
            .round(3)
            .to_string()
        )

# ---------- CLI ----------

def parse_params(param_list):
    out = {}
    for item in param_list or []:
        if "=" not in item:
            continue
        k, v = item.split("=", 1)
        k = k.strip(); v = v.strip()
        try:
            if "." in v:
                v_cast = float(v)
            else:
                v_cast = int(v)
            out[k] = v_cast
        except ValueError:
            if v.lower() in ("true", "false"):
                out[k] = (v.lower() == "true")
            else:
                out[k] = v
    return out

def main():
    ap = argparse.ArgumentParser(description="Audio Noise Reduction Pipeline (chunked/full, with normalization)")
    ap.add_argument("--input_dir", type=Path, required=True, help="Folder with raw audio (recursively scanned)")
    ap.add_argument("--output_dir", type=Path, required=True, help="Folder for cleaned audio & reports")
    ap.add_argument("--method", type=str, default="spectral_gate", choices=["spectral_gate", "wiener"], help="Denoising method")
    ap.add_argument("--sr", type=int, default=16000, help="Target sample rate")

    ap.add_argument("--pilot_count", type=int, default=0, help="Run pilot on first N files before full batch (0=skip)")
    ap.add_argument("--num_workers", type=int, default=max(1, cpu_count() // 2), help="Parallel workers for full batch")

    ap.add_argument("--frame_length", type=int, default=2048, help="Frame length for metrics")
    ap.add_argument("--hop_length", type=int, default=512, help="Hop length for metrics")

    ap.add_argument("--param", action="append", help="Method param key=val; repeatable (e.g., --param prop_decrease=0.9)")

    # Chunked mode
    ap.add_argument("--chunked", action="store_true", help="Enable streaming chunked processing")
    ap.add_argument("--chunk_duration_s", type=float, default=10.0, help="Chunk duration in seconds")
    ap.add_argument("--chunk_overlap_s", type=float, default=0.2, help="Crossfade overlap in seconds")

    # Normalization
    ap.add_argument("--normalize", type=str, default="none", choices=["none", "peak", "rms", "lufs"],
                    help="Normalization mode")
    ap.add_argument("--target_dbfs", type=float, default=-1.0,
                    help="Target dBFS for peak/rms normalization (e.g., -1.0 for peak, -20.0 for rms speech)")
    ap.add_argument("--target_lufs", type=float, default=-23.0,
                    help="Target LUFS for loudness normalization (requires pyloudnorm)")

    ap.add_argument("--writer_subtype", type=str, default="PCM_16", help="SoundFile subtype (PCM_16, PCM_24, FLOAT)")
    ap.add_argument("--save_intermediate", action="store_true", help="Save debug npz with noise stats")

    args = ap.parse_args()
    args.input_dir = args.input_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.method_params = parse_params(args.param)

    files_all = list_audio_files(args.input_dir)
    if not files_all:
        print(f"No audio files found in: {args.input_dir}")
        sys.exit(1)

    # Choose processor
    if args.chunked:
        processor_func = process_one_chunked
        param_tag_extra = {
            "chunk_duration_s": args.chunk_duration_s,
            "chunk_overlap_s": args.chunk_overlap_s,
        }
    else:
        processor_func = process_one_full
        param_tag_extra = {"param_tag_extra": None}

    # Pilot (optional)
    if args.pilot_count > 0:
        pilot_files = files_all[: args.pilot_count]
        run_pilot(pilot_files, args, processor_func, param_tag_extra if args.chunked else None)

    # Full batch
    run_full(files_all, args, processor_func, param_tag_extra if args.chunked else None)

if __name__ == "__main__":
    main()
