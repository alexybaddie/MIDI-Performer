# src/train.py
import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import MidiPairDataset
from model import Encoder, Decoder, Seq2Seq

# -------------------------
# Hyperparameters and paths
# -------------------------
DATA_ROOT = os.path.join("..", "data")  # Adjust path if needed.
BATCH_SIZE = 16
MAX_SEQ_LEN = 500
NUM_EPOCHS = 50
LEARNING_RATE = 0.001

# Dimensions: 
# Score features: [onset, pitch, duration] -> input_dim = 3
# Performance target: [onset, velocity] -> output_dim = 2
INPUT_DIM = 3
OUTPUT_DIM = 2
HIDDEN_DIM = 128
NUM_LAYERS = 2

# -------------------------
# Device configuration
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------------
# Load Dataset and DataLoader
# -------------------------
dataset = MidiPairDataset(data_root=DATA_ROOT, max_seq_len=MAX_SEQ_LEN)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# -------------------------
# Initialize Model
# -------------------------
encoder = Encoder(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS)
decoder = Decoder(output_dim=OUTPUT_DIM, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS)
model = Seq2Seq(encoder, decoder, device).to(device)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.MSELoss()  # Using MSE loss for continuous output

# -------------------------
# Training Loop
# -------------------------
model.train()
for epoch in range(1, NUM_EPOCHS + 1):
    epoch_loss = 0
    for i, (src, trg) in enumerate(dataloader):
        src = src.to(device)  # shape: (batch, seq_len, 3)
        trg = trg.to(device)  # shape: (batch, seq_len, 2)
        
        optimizer.zero_grad()
        output = model(src, trg, teacher_forcing_ratio=0.5)
        loss = criterion(output, trg)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(dataloader)
    print(f"Epoch {epoch}/{NUM_EPOCHS}, Average Loss: {avg_loss:.4f}")

# Save the trained model
MODEL_SAVE_PATH = os.path.join("..", "model.pth")
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"Model saved to {MODEL_SAVE_PATH}")
