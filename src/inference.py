# src/inference.py
import os
import torch
import numpy as np
import pretty_midi
import argparse
from dataset import load_midi_events
from model import Encoder, Decoder, Seq2Seq

# -------------------------
# Hyperparameters (should match training)
# -------------------------
MAX_SEQ_LEN = 500
INPUT_DIM = 3    # [onset, pitch, duration]
OUTPUT_DIM = 2   # [onset, velocity]
HIDDEN_DIM = 128
NUM_LAYERS = 2

# -------------------------
# Device configuration
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_model(model_path):
    encoder = Encoder(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS)
    decoder = Decoder(output_dim=OUTPUT_DIM, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS)
    model = Seq2Seq(encoder, decoder, device).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def pad_or_truncate_array(arr, max_len):
    current_len = arr.shape[0]
    if current_len > max_len:
        return arr[:max_len, :]
    elif current_len < max_len:
        pad_width = ((0, max_len - current_len), (0, 0))
        return np.pad(arr, pad_width, mode='constant', constant_values=0)
    else:
        return arr

def generate_performance(model, score_midi_path, output_midi_path):
    # Load score MIDI events. For input, we ignore velocity.
    # NOTE: If the user-input MIDI is in a different key or has multiple tracks,
    # consider adding a pre-processing step here to (a) merge tracks and (b) transpose to a canonical key.
    score_events = load_midi_events(score_midi_path, include_velocity=False)
    score_arr = np.array(score_events)  # shape: (num_events, 3)
    score_arr = pad_or_truncate_array(score_arr, MAX_SEQ_LEN)
    src = torch.tensor(score_arr, dtype=torch.float32).unsqueeze(0).to(device)  # shape: (1, seq_len, 3)
    
    # For inference, create a dummy target of zeros (will be replaced step-by-step).
    dummy_trg = torch.zeros(1, MAX_SEQ_LEN, OUTPUT_DIM).to(device)
    
    with torch.no_grad():
        predicted = model(src, dummy_trg, teacher_forcing_ratio=0.0)
    # predicted shape: (1, MAX_SEQ_LEN, OUTPUT_DIM)
    predicted = predicted.squeeze(0).cpu().numpy()
    
    # Now, combine the original score’s pitch and duration with the predicted onset and velocity.
    # (For simplicity, we assume the model outputs absolute onset times.)
    # NOTE: Since the model was trained on a dataset in a certain key, if the input score is transposed,
    # the performance output may be less accurate. For better generalization, consider adding key normalization.
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)  # e.g., piano
    for i, event in enumerate(score_events):
        if i >= MAX_SEQ_LEN:
            break
        score_onset, pitch, duration = event
        pred_onset, pred_velocity = predicted[i]
        # You can choose to add the predicted deviation to the original onset:
        final_onset = pred_onset  # or score_onset + pred_onset if the model learns deviations.
        velocity = int(np.clip(pred_velocity, 0, 127))
        note = pretty_midi.Note(velocity=velocity, pitch=int(pitch),
                                start=final_onset, end=final_onset + duration)
        instrument.notes.append(note)
    
    pm.instruments.append(instrument)
    pm.write(output_midi_path)
    print(f"Generated performance saved to {output_midi_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a humanized MIDI performance from a score.")
    parser.add_argument("--model_path", type=str, default=os.path.join("..", "model.pth"),
                        help="Path to the trained model file.")
    parser.add_argument("--input_midi", type=str, required=True,
                        help="Path to the input score MIDI file.")
    parser.add_argument("--output_midi", type=str, default="generated_performance.mid",
                        help="Path to save the generated performance MIDI file.")
    args = parser.parse_args()
    
    model = load_model(args.model_path)
    generate_performance(model, args.input_midi, args.output_midi)
