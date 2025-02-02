# src/dataset.py
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import pretty_midi

def load_midi_events(midi_path, include_velocity=False):
    """
    Reads a MIDI file, merges all tracks, and returns a sorted list of note events.
    
    Each event is represented as a tuple:
       (onset, pitch, duration)  if include_velocity is False
       (onset, pitch, duration, velocity) if include_velocity is True

    Parameters:
      midi_path (str): Path to the MIDI file.
      include_velocity (bool): Whether to include velocity.
      
    Returns:
      List of tuples sorted by onset time.
    """
    try:
        pm = pretty_midi.PrettyMIDI(midi_path)
    except Exception as e:
        raise ValueError(f"Error reading MIDI file {midi_path}: {e}")
    
    events = []
    for instrument in pm.instruments:
        # (Optional) You can filter out drums: if instrument.is_drum: continue
        for note in instrument.notes:
            duration = note.end - note.start
            if include_velocity:
                events.append((note.start, note.pitch, duration, note.velocity))
            else:
                events.append((note.start, note.pitch, duration))
    # Sort events by onset time
    events.sort(key=lambda x: x[0])
    return events

class MidiPairDataset(Dataset):
    """
    Expects a data root directory where each piece is a subdirectory containing:
       - score.mid (non-live, quantized)
       - performance.mid (live performance)
    """
    def __init__(self, data_root, max_seq_len=500):
        self.data_root = data_root
        self.max_seq_len = max_seq_len
        
        # List all piece directories (each must contain score.mid and performance.mid)
        self.piece_dirs = []
        for item in os.listdir(data_root):
            full_path = os.path.join(data_root, item)
            if os.path.isdir(full_path):
                score_file = os.path.join(full_path, "score.mid")
                perf_file = os.path.join(full_path, "performance.mid")
                if os.path.exists(score_file) and os.path.exists(perf_file):
                    self.piece_dirs.append(full_path)
        if not self.piece_dirs:
            raise ValueError("No valid piece directories found in the data root.")
    
    def __len__(self):
        return len(self.piece_dirs)
    
    def __getitem__(self, idx):
        piece_dir = self.piece_dirs[idx]
        score_path = os.path.join(piece_dir, "score.mid")
        perf_path  = os.path.join(piece_dir, "performance.mid")
        
        # Load events. For the score, we ignore velocity.
        score_events = load_midi_events(score_path, include_velocity=False)
        perf_events  = load_midi_events(perf_path, include_velocity=True)
        
        # IMPORTANT: In a real system, you should align events between score and performance.
        # Here we assume the events are in order and correspond one-to-one.
        if len(score_events) != len(perf_events):
            raise ValueError(f"Mismatch in number of events between {score_path} and {perf_path}")
        
        # Convert events to numpy arrays.
        # Score: (num_events, 3): [onset, pitch, duration]
        score_arr = np.array(score_events)
        # Performance: (num_events, 4): [onset, pitch, duration, velocity]
        perf_arr = np.array(perf_events)
        
        # For training, we assume that pitch and duration are “copied” from the score.
        # We want the model to learn the expressive parts: the (onset, velocity) differences.
        # For simplicity, we let the input be the score events and the target be the performance’s
        # onset and velocity.
        src = score_arr  # shape: (num_events, 3)
        trg = perf_arr[:, [0, 3]]  # shape: (num_events, 2) -> [onset, velocity]
        
        # Pad (or truncate) to fixed max_seq_len.
        src = self._pad_or_truncate(src, self.max_seq_len)
        trg = self._pad_or_truncate(trg, self.max_seq_len)
        
        # Convert to torch tensors.
        src = torch.tensor(src, dtype=torch.float32)
        trg = torch.tensor(trg, dtype=torch.float32)
        
        return src, trg

    def _pad_or_truncate(self, arr, max_len):
        current_len = arr.shape[0]
        if current_len > max_len:
            return arr[:max_len, :]
        elif current_len < max_len:
            pad_width = ((0, max_len - current_len), (0, 0))
            return np.pad(arr, pad_width, mode='constant', constant_values=0)
        else:
            return arr
