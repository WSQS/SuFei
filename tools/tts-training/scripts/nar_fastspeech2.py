"""Minimal FastSpeech 2 model in PyTorch.

Architecture (simplified FastSpeech 2):
- Phoneme embedding: vocab_size → d_model
- 4 Transformer encoder layers (d_model=256, nhead=2)
- Duration predictor: 2 conv1d + linear
- Length regulator: expand by durations
- Pitch predictor: 2 conv1d + linear (predicts log F0)
- Energy predictor: 2 conv1d + linear (predicts log energy)
- 4 Transformer decoder layers
- Mel linear: d_model → n_mels

For the overfit test we keep it small but functional.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TransformerBlock(nn.Module):
    def __init__(self, d_model=256, nhead=2, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=mask)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.linear2(self.dropout(F.relu(self.linear1(x))))
        x = self.norm2(x + self.dropout(ff_out))
        return x


class VariancePredictor(nn.Module):
    def __init__(self, d_model=256, filter_size=256, kernel_size=3, dropout=0.5):
        super().__init__()
        self.conv1 = nn.Conv1d(d_model, filter_size, kernel_size, padding=kernel_size // 2)
        self.norm1 = nn.LayerNorm(filter_size)
        self.conv2 = nn.Conv1d(filter_size, filter_size, kernel_size, padding=kernel_size // 2)
        self.norm2 = nn.LayerNorm(filter_size)
        self.linear = nn.Linear(filter_size, 1)
        self.dropout = nn.Dropout(dropout)

    def init_bias(self, value):
        """Initialize final linear bias to target value, tiny weights.
        
        Bias ≈ log(mean_dur) gives correct baseline. Weight ≈ 1e-4
        allows gradient flow while keeping initial predictions near bias.
        """
        torch.nn.init.constant_(self.linear.bias, value)
        torch.nn.init.normal_(self.linear.weight, mean=0.0, std=1e-4)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.dropout(F.relu(self.conv1(x)))
        x = x.transpose(1, 2)
        x = self.norm1(x)
        x = x.transpose(1, 2)
        x = self.dropout(F.relu(self.conv2(x)))
        x = x.transpose(1, 2)
        x = self.norm2(x)
        x = self.dropout(self.linear(x))
        return x.squeeze(-1)


class LengthRegulator(nn.Module):
    def forward(self, x, durations):
        """Expand encoder hidden states by durations (vectorized).

        x: [B, L, D]
        durations: [B, L] int

        Returns: [B, T_max, D] padded
        """
        B, L, D = x.shape
        device = x.device

        # Clamp durations to >= 0
        durations = durations.clamp(min=0)
        total_durs = durations.sum(dim=1)
        T_max = int(total_durs.max().item())
        if T_max == 0:
            T_max = 1

        # Cumulative durations give frame boundaries
        cumdurs = torch.cumsum(durations, dim=1)  # [B, L]
        # For each output frame t in [0, T_max), find which phoneme it belongs to
        # frame_indices: [T_max]
        frame_indices = torch.arange(T_max, device=device).unsqueeze(0)  # [1, T_max]
        # For each (b, t), find the smallest i such that cumdurs[b, i] > t
        # That's the phoneme index for frame t
        # mask[b, i, t] = True if frame t belongs to phoneme i
        # cumdurs[b, i] > frame_indices[t] means frame t is before the end of phoneme i
        # We want the first i where this is true
        phoneme_per_frame = (
            (cumdurs.unsqueeze(2) > frame_indices.unsqueeze(1))  # [B, L, T_max]
            .float()
            .argmax(dim=1)  # [B, T_max] — first True along L
        )

        # Clamp to valid range [0, L-1]
        phoneme_per_frame = phoneme_per_frame.clamp(max=L - 1)

        # Gather: output[b, t] = x[b, phoneme_per_frame[b, t]]
        output = torch.gather(
            x, 1, phoneme_per_frame.unsqueeze(-1).expand(-1, -1, D)
        )  # [B, T_max, D]

        return output


class FastSpeech2(nn.Module):
    def __init__(
        self,
        vocab_size=268,
        d_model=256,
        nhead=2,
        num_encoder_layers=4,
        num_decoder_layers=4,
        dim_feedforward=1024,
        n_mels=80,
        max_len=5000,
        dropout=0.1,
        predictor_dropout=0.1,
        mean_log_dur=2.7,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_mels = n_mels

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_enc = PositionalEncoding(d_model, max_len)

        # Encoder
        self.encoder_layers = nn.ModuleList([
            TransformerBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_encoder_layers)
        ])

        # Variance predictors
        self.duration_predictor = VariancePredictor(d_model, dropout=predictor_dropout)
        self.duration_predictor.init_bias(mean_log_dur)
        self.pitch_predictor = VariancePredictor(d_model, dropout=predictor_dropout)
        self.energy_predictor = VariancePredictor(d_model, dropout=predictor_dropout)

        # Length regulator
        self.length_regulator = LengthRegulator()

        # Pitch & energy embedding (broadcast scalar → vector)
        self.pitch_embed = nn.Linear(1, d_model)
        self.energy_embed = nn.Linear(1, d_model)

        # Decoder
        self.decoder_layers = nn.ModuleList([
            TransformerBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_decoder_layers)
        ])

        # Mel output
        self.mel_linear = nn.Linear(d_model, n_mels)

    def forward(self, phoneme_ids, durations=None, pitches=None, energies=None,
                phone_mask=None):
        """
        Args:
            phoneme_ids: [B, L]
            durations: [B, L] in mel frames (ground truth for teacher-forcing)
            pitches: [B, T_max] ground truth F0 (teacher-forcing)
            energies: [B, T_max] ground truth energy (teacher-forcing)
            phone_mask: [B, L] True=padding

        Returns:
            mel_output: [B, T, n_mels]
            log_pred_durations: [B, L]
            pred_pitches_enc: [B, L]
            pred_energies_enc: [B, L]
        """
        x = self.embedding(phoneme_ids) * math.sqrt(self.d_model)
        x = self.pos_enc(x)

        for layer in self.encoder_layers:
            x = layer(x, mask=phone_mask)

        # Predict variance
        log_pred_durations = self.duration_predictor(x)

        pred_pitches_enc = self.pitch_predictor(x)
        pred_energies_enc = self.energy_predictor(x)

        # Length regulate
        if durations is None:
            durations = log_pred_durations.detach().exp().round().clamp(min=0).long()

        mel_input = self.length_regulator(x, durations)

        # Expand pitch/energy predictions to mel length
        expanded_pitches_enc = self.length_regulator(
            pred_pitches_enc.unsqueeze(-1), durations
        ).squeeze(-1)
        expanded_energies_enc = self.length_regulator(
            pred_energies_enc.unsqueeze(-1), durations
        ).squeeze(-1)

        # Use predicted or ground truth pitch/energy
        T_out = mel_input.size(1)
        if pitches is not None:
            T_p = min(pitches.size(1), T_out)
            pitch_embed = self.pitch_embed(pitches[:, :T_out].unsqueeze(-1))
        else:
            pitch_embed = self.pitch_embed(expanded_pitches_enc.unsqueeze(-1))

        if energies is not None:
            energy_embed = self.energy_embed(energies[:, :T_out].unsqueeze(-1))
        else:
            energy_embed = self.energy_embed(expanded_energies_enc.unsqueeze(-1))

        # Truncate or pad mel_input to match pitch/energy embed
        mel_input = mel_input[:, :T_out] + pitch_embed[:, :T_out] + energy_embed[:, :T_out]
        mel_input = self.pos_enc(mel_input)

        # Decode
        dec = mel_input
        for layer in self.decoder_layers:
            dec = layer(dec)

        mel_output = self.mel_linear(dec)
        return mel_output, log_pred_durations, pred_pitches_enc, pred_energies_enc
