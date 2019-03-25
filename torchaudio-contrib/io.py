import torch
import numpy as np
import warnings
from scipy.io import wavfile
import librosa

def load(filename, sr=44100, mono=True, norm=False, device='cpu', dtype=np.float32):
    """
    Generic wrapper for reading an audio file.
    Different libraries offer different speeds for this, so this routine is the
    'catch-all' for whatever read routine happens to work best

    Tries a fast method via scipy first, reverts to slower librosa when necessary.
    """
    # first try to read via scipy, because it's fast
    scipy_ok = False
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("error")    # scipy throws warnings which should be errors
        try:
            out_sr, signal = wavfile.read(filename)
            scipy_ok = True
        except wavfile.WavFileWarning:
            print("read_audio_file: Warning raised by scipy. ",end="")

    if scipy_ok:
        if mono and (len(signal.shape) > 1):     # convert to mono (by truncating other channels)
            signal = signal[:,0]

        if isinstance(signal[0], np.int16):      # convert from ints to floats if necessary
            signal = np.array(signal, dtype=dtype)/32767.0   # change from [-32767..32767] to [-1..1]

        if out_sr != int(sr):
            print(f"read_audio_file: Got sample rate of {rate} Hz instead of {sr} Hz requested. Resampling.")
            signal = librosa.resample(signal, rate*1.0, sr*1.0, res_type='kaiser_fast')

    else:                                         # try librosa; it's slower but general
        print("Trying librosa.")
        signal, out_sr = librosa.core.load(filename, mono=mono, sr=sr, res_type='kaiser_fast')

    if norm:
        signal = signal/np.max(np.abs(signal))

    return torch.from_numpy(signal.astype(dtype)).to(device), out_sr


def save(filename, data, sr=44100):
    #TODO: add support for non-WAV file formats
    wavfile.write(filename, sr, data.cpu().numpy())
