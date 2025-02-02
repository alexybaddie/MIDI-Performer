# src/model.py
import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        """
        Encoder using an LSTM.
        
        Parameters:
          input_dim (int): Dimensionality of input features (e.g. 3 for [onset, pitch, duration])
          hidden_dim (int): Number of hidden units.
          num_layers (int): Number of LSTM layers.
        """
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
    
    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        outputs, (hidden, cell) = self.lstm(x)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, num_layers):
        """
        Decoder using an LSTM.
        
        Parameters:
          output_dim (int): Dimensionality of output features (e.g. 2 for [onset, velocity])
          hidden_dim (int): Number of hidden units.
          num_layers (int): Number of LSTM layers.
        """
        super(Decoder, self).__init__()
        self.lstm = nn.LSTM(output_dim, hidden_dim, num_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, input_step, hidden, cell):
        # input_step: (batch, 1, output_dim)
        output, (hidden, cell) = self.lstm(input_step, (hidden, cell))
        prediction = self.fc_out(output)  # (batch, 1, output_dim)
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        """
        A sequence-to-sequence model combining an encoder and a decoder.
        """
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
    
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        """
        Parameters:
          src: Tensor of shape (batch, src_seq_len, input_dim)
          trg: Tensor of shape (batch, trg_seq_len, output_dim)
          teacher_forcing_ratio: The probability to use teacher forcing.
          
        Returns:
          outputs: Tensor of shape (batch, trg_seq_len, output_dim)
        """
        batch_size = src.size(0)
        trg_len = trg.size(1)
        output_dim = trg.size(2)
        
        # Tensor to store decoder outputs
        outputs = torch.zeros(batch_size, trg_len, output_dim).to(self.device)
        
        # Encode the source sequence
        hidden, cell = self.encoder(src)
        
        # Use the first target token as the initial input for the decoder.
        # (This could be a learned <SOS> token in a more advanced system.)
        input_dec = trg[:, 0, :].unsqueeze(1)  # shape: (batch, 1, output_dim)
        outputs[:, 0, :] = input_dec.squeeze(1)
        
        # Decode the remainder of the sequence one time step at a time.
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input_dec, hidden, cell)
            outputs[:, t, :] = output.squeeze(1)
            
            # Decide whether to use teacher forcing.
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            input_dec = trg[:, t, :].unsqueeze(1) if teacher_force else output
        
        return outputs
