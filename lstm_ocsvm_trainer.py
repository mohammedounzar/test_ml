"""
LSTM-OC-SVM Training Algorithm Implementation
Based on "Unsupervised and Semi-supervised Anomaly Detection with LSTM Neural Networks"
by Tolga Ergen, Ali H. Mirza, and Suleyman S. Kozat

This file implements Algorithm 2: Gradient Based Training for LSTM-OC-SVM
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader


class LSTMEncoder(nn.Module):
    """
    LSTM Encoder as described in the paper (Section II, Equations 1-6)
    Converts variable length sequences to fixed length representations
    """
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(LSTMEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM parameters as described in equations (1)-(6)
        # W^(z), W^(s), W^(f), W^(o) for input transformations
        self.W_z = nn.Parameter(torch.randn(hidden_size, input_size))
        self.W_s = nn.Parameter(torch.randn(hidden_size, input_size))
        self.W_f = nn.Parameter(torch.randn(hidden_size, input_size))
        self.W_o = nn.Parameter(torch.randn(hidden_size, input_size))
        
        # R^(z), R^(s), R^(f), R^(o) for recurrent transformations
        self.R_z = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.R_s = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.R_f = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.R_o = nn.Parameter(torch.randn(hidden_size, hidden_size))
        
        # Bias terms b^(z), b^(s), b^(f), b^(o)
        self.b_z = nn.Parameter(torch.randn(hidden_size))
        self.b_s = nn.Parameter(torch.randn(hidden_size))
        self.b_f = nn.Parameter(torch.randn(hidden_size))
        self.b_o = nn.Parameter(torch.randn(hidden_size))
        
        # Initialize with orthogonality constraints (Equation 9)
        self._initialize_orthogonal()
    
    def _initialize_orthogonal(self):
        """Initialize parameters with orthogonality constraints as per Equation (9)"""
        # W^(·)^T W^(·) = I
        for W in [self.W_z, self.W_s, self.W_f, self.W_o]:
            nn.init.orthogonal_(W)
        
        # R^(·)^T R^(·) = I  
        for R in [self.R_z, self.R_s, self.R_f, self.R_o]:
            nn.init.orthogonal_(R)
        
        # b^(·)^T b^(·) = 1
        for b in [self.b_z, self.b_s, self.b_f, self.b_o]:
            nn.init.normal_(b)
            b.data = b.data / torch.norm(b.data)
    
    def enforce_orthogonality_constraints(self):
        """
        Enforce orthogonality constraints from Equation (9) during training
        This is crucial for avoiding overfitting and learning time dependencies
        """
        with torch.no_grad():
            # Orthogonalize W matrices
            for W in [self.W_z, self.W_s, self.W_f, self.W_o]:
                U, _, V = torch.svd(W)
                W.copy_(U @ V.t())
            
            # Orthogonalize R matrices
            for R in [self.R_z, self.R_s, self.R_f, self.R_o]:
                U, _, V = torch.svd(R)
                R.copy_(U @ V.t())
            
            # Normalize bias vectors
            for b in [self.b_z, self.b_s, self.b_f, self.b_o]:
                b.copy_(b / torch.norm(b))
    
    def forward(self, x):
        """
        Forward pass implementing LSTM equations (1)-(6) from the paper
        
        Args:
            x: Input sequences of shape (batch_size, seq_len, input_size)
        
        Returns:
            h_bar: Fixed length representations via mean pooling (batch_size, hidden_size)
        """
        batch_size, seq_len, _ = x.shape
        
        # Initialize hidden and cell states
        h = torch.zeros(batch_size, self.hidden_size, device=x.device)
        c = torch.zeros(batch_size, self.hidden_size, device=x.device)
        
        h_outputs = []
        
        # Process each time step according to equations (1)-(6)
        for t in range(seq_len):
            x_t = x[:, t, :]  # Current input
            
            # Equation (1): z_{i,j} = g(W^(z) x_{i,j} + R^(z) h_{i,j-1} + b^(z))
            z = torch.tanh(x_t @ self.W_z.t() + h @ self.R_z.t() + self.b_z)
            
            # Equation (2): s_{i,j} = σ(W^(s) x_{i,j} + R^(s) h_{i,j-1} + b^(s))
            s = torch.sigmoid(x_t @ self.W_s.t() + h @ self.R_s.t() + self.b_s)
            
            # Equation (3): f_{i,j} = σ(W^(f) x_{i,j} + R^(f) h_{i,j-1} + b^(f))
            f = torch.sigmoid(x_t @ self.W_f.t() + h @ self.R_f.t() + self.b_f)
            
            # Equation (4): c_{i,j} = s_{i,j} ⊙ z_{i,j} + f_{i,j} ⊙ c_{i,j-1}
            c = s * z + f * c
            
            # Equation (5): o_{i,j} = σ(W^(o) x_{i,j} + R^(o) h_{i,j-1} + b^(o))
            o = torch.sigmoid(x_t @ self.W_o.t() + h @ self.R_o.t() + self.b_o)
            
            # Equation (6): h_{i,j} = o_{i,j} ⊙ g(c_{i,j})
            h = o * torch.tanh(c)
            
            h_outputs.append(h)
        
        # Mean pooling as described in Section II and Figure 2
        # h̄_i = (1/d_i) Σ_{j=1}^{d_i} h_{i,j}
        h_stacked = torch.stack(h_outputs, dim=1)  # (batch_size, seq_len, hidden_size)
        h_bar = torch.mean(h_stacked, dim=1)  # Mean pooling
        
        return h_bar


class LSTMOCSVMJoint(nn.Module):
    """
    Joint LSTM + OC-SVM model implementing the gradient-based training from Algorithm 2
    """
    def __init__(self, input_size, hidden_size, n_lambda=0.05, tau=10.0):
        super(LSTMOCSVMJoint, self).__init__()
        
        self.lstm = LSTMEncoder(input_size, hidden_size)
        self.n_lambda = n_lambda  # Regularization parameter λ from equation (7)
        self.tau = tau  # Smoothing parameter τ from equation (29)
        
        # OC-SVM parameters from equation (7)
        self.w = nn.Parameter(torch.randn(hidden_size))  # Hyperplane normal vector
        self.rho = nn.Parameter(torch.randn(1))  # Hyperplane offset
        
    def smooth_max(self, beta):
        """
        Smooth approximation S_τ(β) from Equation (29)
        S_τ(β) = (1/τ) log(1 + e^(τβ))
        This approximates max(0, β) for gradient-based optimization
        """
        return (1.0 / self.tau) * torch.log(1 + torch.exp(self.tau * beta))
    
    def forward(self, x):
        """
        Forward pass computing the joint objective function F_τ from Equation (30)
        """
        # Get LSTM representations
        h_bar = self.lstm(x)  # Shape: (batch_size, hidden_size)
        
        # Compute β_{w,ρ}(h̄_i) = ρ - w^T h̄_i for each sample
        beta = self.rho - torch.matmul(h_bar, self.w)  # Shape: (batch_size,)
        
        # Apply smooth approximation S_τ(β) from equation (29)
        smooth_slack = self.smooth_max(beta)  # Shape: (batch_size,)
        
        # Compute objective function F_τ from equation (30)
        # F_τ(w, ρ, θ) = ||w||²/2 + (1/nλ) Σ S_τ(β_{w,ρ}(h̄_i)) - ρ
        regularization = 0.5 * torch.norm(self.w) ** 2
        slack_penalty = torch.mean(smooth_slack) / self.n_lambda
        offset_penalty = -self.rho
        
        loss = regularization + slack_penalty + offset_penalty
        
        return loss, h_bar
    
    def predict(self, x):
        """
        Anomaly detection using scoring function from Equation (10)
        l(X_i) = sgn(w^T h̄_i - ρ)
        Returns +1 for normal, -1 for anomaly
        """
        with torch.no_grad():
            h_bar = self.lstm(x)
            scores = torch.matmul(h_bar, self.w) - self.rho
            predictions = torch.sign(scores)
        return predictions, scores


class VariableLengthDataset(Dataset):
    """Dataset for variable length sequences as mentioned in the paper"""
    def __init__(self, sequences):
        self.sequences = sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.float32)


def collate_variable_length(batch):
    """
    Collate function to handle variable length sequences
    Pads sequences to the same length for batch processing
    """
    # Find maximum length in the batch
    max_len = max(seq.shape[0] for seq in batch)
    
    # Pad sequences
    padded_batch = []
    for seq in batch:
        seq_len, input_size = seq.shape
        if seq_len < max_len:
            padding = torch.zeros(max_len - seq_len, input_size)
            padded_seq = torch.cat([seq, padding], dim=0)
        else:
            padded_seq = seq
        padded_batch.append(padded_seq)
    
    return torch.stack(padded_batch)


class LSTMOCSVMTrainer:
    """
    Trainer implementing Algorithm 2: Gradient Based Training for LSTM-OC-SVM
    """
    def __init__(self, model, learning_rate=0.001, orthogonal_update_freq=10):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.orthogonal_update_freq = orthogonal_update_freq
        self.iteration = 0
    
    def train_step(self, batch):
        """
        Single training step implementing the gradient updates from Algorithm 2
        """
        self.optimizer.zero_grad()
        
        # Forward pass - compute F_τ(w, ρ, θ)
        loss, h_bar = self.model(batch)
        
        # Backward pass - compute gradients as in equations (32), (34), (36)
        loss.backward()
        
        # Update parameters using SGD as described in equations (33), (35), (37)
        self.optimizer.step()
        
        # Enforce orthogonality constraints periodically (equation 9)
        if self.iteration % self.orthogonal_update_freq == 0:
            self.model.lstm.enforce_orthogonality_constraints()
        
        self.iteration += 1
        
        return loss.item()
    
    def train(self, dataloader, epochs, convergence_threshold=1e-6):
        """
        Complete training loop implementing Algorithm 2
        """
        self.model.train()
        prev_loss = float('inf')
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            for batch in dataloader:
                loss = self.train_step(batch)
                epoch_loss += loss
            
            avg_loss = epoch_loss / len(dataloader)
            
            # Check convergence as in Algorithm 2, line 8
            if abs(prev_loss - avg_loss) < convergence_threshold:
                print(f"Converged at epoch {epoch+1}")
                break
            
            prev_loss = avg_loss
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
        
        return self.model


def train_lstm_ocsvm(sequences, input_size, hidden_size=64, n_lambda=0.05, tau=10.0, 
                    batch_size=16, epochs=100, learning_rate=0.001, 
                    convergence_threshold=1e-6, orthogonal_update_freq=10):
    """
    Main training function implementing the LSTM-OC-SVM algorithm from the paper
    
    Args:
        sequences: List of variable-length sequences (numpy arrays)
        input_size: Dimensionality of input features (p in the paper)
        hidden_size: LSTM hidden dimension (m in the paper)
        n_lambda: Regularization parameter λ from equation (7)
        tau: Smoothing parameter τ from equation (29)
        batch_size: Batch size for training
        epochs: Maximum number of training epochs
        learning_rate: Learning rate for gradient updates
        convergence_threshold: Convergence criterion (ε in Algorithm 2)
        orthogonal_update_freq: Frequency of orthogonality constraint enforcement
    
    Returns:
        trained_model: Trained LSTM-OC-SVM model
        trainer: Trainer object (for further training if needed)
    """
    
    print("Initializing LSTM-OC-SVM model...")
    print(f"Input size: {input_size}, Hidden size: {hidden_size}")
    print(f"Lambda: {n_lambda}, Tau: {tau}")
    
    # Create dataset and dataloader
    dataset = VariableLengthDataset(sequences)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                          collate_fn=collate_variable_length)
    
    # Initialize model following the paper's specifications
    model = LSTMOCSVMJoint(input_size, hidden_size, n_lambda, tau)
    
    # Initialize trainer with Algorithm 2 parameters
    trainer = LSTMOCSVMTrainer(model, learning_rate, orthogonal_update_freq)
    
    print(f"Starting training with {len(sequences)} sequences...")
    print(f"Training parameters: epochs={epochs}, lr={learning_rate}, batch_size={batch_size}")
    
    # Train the model using Algorithm 2
    trained_model = trainer.train(dataloader, epochs, convergence_threshold)
    
    print("Training completed!")
    
    return trained_model, trainer


def predict_anomalies(model, sequences):
    """
    Use trained model to predict anomalies
    
    Args:
        model: Trained LSTM-OC-SVM model
        sequences: List of sequences to classify
    
    Returns:
        predictions: Array of predictions (+1 normal, -1 anomaly)
        scores: Array of anomaly scores
    """
    model.eval()
    
    dataset = VariableLengthDataset(sequences)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, 
                          collate_fn=collate_variable_length)
    
    all_predictions = []
    all_scores = []
    
    with torch.no_grad():
        for batch in dataloader:
            predictions, scores = model.predict(batch)
            all_predictions.append(predictions.cpu().numpy())
            all_scores.append(scores.cpu().numpy())
    
    return np.concatenate(all_predictions), np.concatenate(all_scores)