import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch

from argparse import ArgumentParser
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, roc_auc_score
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from tools.utils import seed_everything, str2bool, load_data, save_metrics, refine_dict


class Model(nn.Module):
    """Standard neural network for point prediction."""
    def __init__(self, input_size, hidden_dim_str):
        super(Model, self).__init__()
        hidden_dims = [input_size] + list(map(int, hidden_dim_str.split(',')))
        self.layers = nn.ModuleList(
            nn.Linear(hidden_dims[i], hidden_dims[i + 1]) for i in range(len(hidden_dims) - 1)
        )
        self.output_layer = nn.Linear(hidden_dims[-1], 1)

    def forward(self, x):
        for layer in self.layers:
            x = F.leaky_relu(layer(x))
        x = self.output_layer(x)
        return x


class ConformalPredictor:
    """
    Adaptive Conformal Inference (ACI) for handling distribution shift.
    
    This class implements ACI which adapts the conformal threshold online to maintain
    long-term coverage guarantees under distribution shift.
    
    Reference:
    Gibbs, I., & Candès, E. (2021). 
    Adaptive conformal inference under distribution shift. NeurIPS 2021.
    """
    def __init__(self, model, alpha=0.1, gamma=0.005):
        """
        Args:
            model: The trained reward model
            alpha: Target miscoverage rate (default 0.1 for 90% coverage)
            gamma: Learning rate for threshold updates (default 0.005)
        """
        self.model = model
        self.alpha = alpha
        self.gamma = gamma
        self.q_t = None  # Adaptive threshold (will be initialized during calibration)
        self.err_t = 0.0  # Cumulative coverage error
        
    def calibrate(self, X_cal, y_cal, binary=True):
        """
        Initialize the adaptive threshold using calibration set.
        
        Args:
            X_cal: Calibration features
            y_cal: Calibration labels
            binary: Whether using binary classification
            
        Returns:
            q_0: Initial threshold
        """
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_cal).squeeze()
            
            if binary:
                # For binary classification, use probability predictions
                probs = torch.sigmoid(outputs)
                # Compute nonconformity scores as 1 - probability of true class
                scores = torch.where(
                    y_cal > 0.5,
                    1 - probs,  # For positive class
                    probs       # For negative class
                )
            else:
                # For regression, use absolute residuals
                scores = torch.abs(outputs - y_cal)
        
        # Initialize q_t with standard conformal quantile
        n = len(scores)
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        self.q_t = torch.quantile(scores, q_level).item()
        
        # Initialize error to 0
        self.err_t = 0.0
        
        return self.q_t
    
    def reset_to_threshold(self, q_init, err_init=0.0):
        """
        Reset the predictor to a specific threshold and error state.
        
        This is useful for:
        1. Deploying on new data stream starting from q_0
        2. Continuing from where test set left off (q_T)
        3. Reusing a previously calibrated predictor
        
        Args:
            q_init: Initial threshold to start from (e.g., q_0 or q_T)
            err_init: Initial cumulative error (default: 0.0 for fresh start)
            
        Example:
            # For new data similar to calibration set
            predictor.reset_to_threshold(q_0)
            
            # For new data similar to test set
            predictor.reset_to_threshold(q_T, err_T)
        """
        self.q_t = q_init
        self.err_t = err_init
        print(f"Predictor reset: q_t={q_init:.4f}, err_t={err_init:.4f}")
    
    def update_threshold(self, coverage_indicator):
        """
        Update the adaptive threshold based on observed coverage.
        
        Args:
            coverage_indicator: 1 if true label is covered, 0 otherwise
            
        This implements the PID-like update rule:
            err_t = err_{t-1} + alpha - coverage_indicator
            q_t = q_{t-1} + gamma * err_t
        """
        # Update cumulative error
        self.err_t = self.err_t + self.alpha - coverage_indicator
        
        # Update threshold (with clipping to ensure it stays in valid range)
        self.q_t = max(0.0, min(1.0, self.q_t + self.gamma * self.err_t))
        
    def predict_with_uncertainty(self, X, binary=True):
        """
        Make predictions with adaptive conformal prediction intervals.
        
        Args:
            X: Input features
            binary: Whether using binary classification
            
        Returns:
            predictions: Point predictions
            lower_bounds: Lower bounds of prediction sets
            upper_bounds: Upper bounds of prediction sets
            set_sizes: Sizes of prediction sets
        """
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X).squeeze()
            
            if binary:
                probs = torch.sigmoid(outputs)
                predictions = probs
                
                # Construct prediction sets
                # A label is included if 1 - P(label) <= q_t
                # Equivalently, P(label) >= 1 - q_t
                threshold = 1 - self.q_t
                
                # Check which labels are in the prediction set
                pred_set_positive = (probs >= threshold).float()
                pred_set_negative = ((1 - probs) >= threshold).float()
                
                # Set sizes: number of labels in prediction set
                set_sizes = pred_set_positive + pred_set_negative
                
                # For binary, we represent uncertainty by the prediction set
                # Lower and upper bounds represent the range
                lower_bounds = torch.where(pred_set_negative > 0,
                                          torch.zeros_like(probs),
                                          probs - self.q_t)
                upper_bounds = torch.where(pred_set_positive > 0,
                                          torch.ones_like(probs),
                                          probs + self.q_t)
            else:
                # For regression
                predictions = outputs
                lower_bounds = outputs - self.q_t
                upper_bounds = outputs + self.q_t
                set_sizes = (upper_bounds - lower_bounds).abs()
        
        return predictions, lower_bounds, upper_bounds, set_sizes
    
    def get_coverage_indicator(self, X, y, binary=True):
        """
        Check if true label is covered by prediction set for a single sample.
        
        Args:
            X: Input features (single sample or batch)
            y: True labels (single sample or batch)
            binary: Whether using binary classification
            
        Returns:
            coverage_indicator: 1 if covered, 0 otherwise (for each sample)
        """
        predictions, lower_bounds, upper_bounds, _ = self.predict_with_uncertainty(X, binary)
        
        if binary:
            # Coverage: true label is in prediction set
            y_binary = (y > 0.5).float()
            probs = predictions
            threshold = 1 - self.q_t
            
            coverage_mask = torch.where(
                y_binary > 0.5,
                probs >= threshold,      # Positive label covered
                (1 - probs) >= threshold # Negative label covered
            )
            coverage_indicator = coverage_mask.float()
        else:
            # For regression, check if true value is in interval
            coverage_indicator = ((y >= lower_bounds) & (y <= upper_bounds)).float()
        
        return coverage_indicator
    
    def adaptive_predict_online(self, X_stream, y_stream, binary=True):
        """
        Perform adaptive conformal prediction on a data stream.
        
        This method processes samples sequentially and updates the threshold online.
        
        Args:
            X_stream: Stream of input features
            y_stream: Stream of true labels (revealed after prediction)
            binary: Whether using binary classification
            
        Returns:
            coverages: Coverage at each time step
            set_sizes: Prediction set sizes at each time step
            thresholds: Adaptive threshold at each time step
        """
        n = len(X_stream)
        coverages = []
        set_sizes = []
        thresholds = []
        
        for t in range(n):
            # Get single sample
            X_t = X_stream[t:t+1]
            y_t = y_stream[t:t+1]
            
            # Make prediction with current threshold
            _, _, _, size = self.predict_with_uncertainty(X_t, binary)
            
            # Check coverage
            coverage = self.get_coverage_indicator(X_t, y_t, binary)
            
            # Record metrics
            coverages.append(coverage.item())
            set_sizes.append(size.item())
            thresholds.append(self.q_t)
            
            # Update threshold for next iteration
            self.update_threshold(coverage.item())
        
        return np.array(coverages), np.array(set_sizes), np.array(thresholds)
    
    def get_coverage_and_efficiency(self, X, y, binary=True):
        """
        Evaluate coverage and efficiency of ACI prediction intervals.
        Uses online adaptive updates by default (this is the core of ACI).
        
        Args:
            X: Test features
            y: Test labels
            binary: Whether using binary classification
            
        Returns:
            coverage: Empirical coverage rate
            avg_set_size: Average prediction set size
            final_threshold: Final adaptive threshold
        """
        # Online adaptive evaluation
        coverages, set_sizes, thresholds = self.adaptive_predict_online(X, y, binary)
        coverage = coverages.mean()
        avg_set_size = set_sizes.mean()
        final_threshold = thresholds[-1]
        
        return coverage, avg_set_size, final_threshold


def parse_arguments():
    # Pre-parse only data_name to select dataset defaults
    pre_parser = ArgumentParser(add_help=False)
    pre_parser.add_argument("--data_name", type=str, default="hs")
    pre_args, _ = pre_parser.parse_known_args()

    # Base defaults
    base_defaults = {
        "desc": "aci",
        "is_training": True,
        "output_dir": f"./results/cache/aci/{pre_args.data_name}",
        "data_root": "./embeddings/clean",
        "model_name": "FsfairX-LLaMA3-RM-v0.1",
        "estimator_name": "aci",
        "data_name": pre_args.data_name,
        "lr": 0.0005,
        "num_epochs": 600,
        "batch_size": 512,
        "hidden_dim": "256,64",
        "patience": 30,
        "seed": 42,
        "l2_reg": 1e-6,
        "w_reg": 1.0,
        "rerun": False,
        "binary": True,  # Support both binary and regression
        "use_tqdm": True,
        "alpha": 0.1,  # Target miscoverage rate
        "gamma": 0.005,  # Learning rate for adaptive updates
    }

    dataset_defaults = {
        "hs": {
            "l2_reg": 1e-6,
            "w_reg": 1.0,
            "alpha": 0.1,
            "gamma": 0.005,
        },
        "ufb": {
            "l2_reg": 1e-5,
            "w_reg": 1.0,
            "alpha": 0.1,
            "gamma": 0.005,
        },
        "saferlhf": {
            "l2_reg": 1e-7,
            "w_reg": 1.0,
            "alpha": 0.1,
            "gamma": 0.005,
        },
    }
    ds_defaults = dataset_defaults.get(pre_args.data_name, {})
    merged_defaults = {**base_defaults, **ds_defaults}

    # Full parser
    parser = ArgumentParser(description="Adaptive Conformal Inference for Reward Models")
    parser.add_argument("--desc", type=str)
    parser.add_argument("--is_training", type=str2bool)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--data_root", type=str)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--data_name", type=str)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--num_epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--hidden_dim", type=str, help="Hidden dimensions for model")
    parser.add_argument("--patience", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--l2_reg", type=float, help="L2 regularization coefficient")
    parser.add_argument("--w_reg", type=float, help="Weight for regularization term in loss")
    parser.add_argument("--rerun", type=str2bool, help="Whether to rerun the experiment")
    parser.add_argument("--binary", type=str2bool, help="Whether to use binary labels")
    parser.add_argument("--use_tqdm", type=str2bool, help="Whether to use tqdm progress bar")
    parser.add_argument("--alpha", type=float, help="Target miscoverage rate for ACI")
    parser.add_argument("--gamma", type=float, help="Learning rate for adaptive threshold updates")

    parser.set_defaults(**merged_defaults)
    args = parser.parse_args()
    return args


def train(model, train_loader, optimizer, num_epochs, patience, args):
    """Train the reward model using standard supervised learning."""
    if not args.is_training:
        return

    best_loss = float('inf')
    patience_counter = 0

    criterion = nn.BCEWithLogitsLoss() if args.binary else nn.MSELoss()

    for epoch in range(num_epochs):
        epoch_loss = 0
        model.train()

        bar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{num_epochs}", leave=False) if args.use_tqdm else train_loader
        for batch_X, batch_y in bar:
            optimizer.zero_grad()
            reward_pred = model(batch_X).squeeze()
            loss = criterion(reward_pred, batch_y)
            
            weighted_loss = args.w_reg * loss
            weighted_loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        monitor_loss = epoch_loss / len(train_loader)
        if epoch % 10 == 0:
            print(f'Epoch {epoch + 1}/{num_epochs}, Train loss: {monitor_loss:.5f}')

        if monitor_loss < best_loss:
            best_loss = monitor_loss
            patience_counter = 0
            torch.save(model.state_dict(), f'{args.output_dir}/best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break


def main():
    args = parse_arguments()

    if args.is_training and os.path.exists(f"{args.output_dir}/performance.yaml") and not args.rerun:
        print(f"The path {args.output_dir}/performance.yaml exists!!")
        sys.exit()
    os.makedirs(args.output_dir, exist_ok=True)

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("="*70)
    print("Adaptive Conformal Inference (ACI) for Reward Modeling")
    print("="*70)
    print("Loading preprocessed data from Safetensors file...")
    
    # Load all data from preprocessed file
    data_file = f"{args.data_root}/{args.model_name}_{args.data_name}.safetensors"
    
    if args.binary:
        X_train, y_train, X_cal, y_cal, X_test, y_test = load_data(
            data_file, device,
            keys=["X_train", "y_train_binary", "X_cal", "y_cal_binary", "X_test", "y_test_binary"]
        )
    else:
        X_train, y_train, X_cal, y_cal, X_test, y_test = load_data(
            data_file, device,
            keys=["X_train", "y_train", "X_cal", "y_cal", "X_test", "y_test"]
        )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Calibration set: {X_cal.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Adaptive learning rate (gamma): {args.gamma}")

    # Train reward model
    print("\n" + "="*70)
    print("Step 1: Training Reward Model")
    print("="*70)
    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=args.batch_size,
        shuffle=True
    )
    model = Model(X_train.shape[1], args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_reg)

    train(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        num_epochs=args.num_epochs,
        patience=args.patience,
        args=args
    )
    
    # Load best model
    model.load_state_dict(torch.load(f'{args.output_dir}/best_model.pth'))
    model.eval()

    # ACI Calibration
    print("\n" + "="*70)
    print("Step 2: Initializing Adaptive Conformal Predictor")
    print("="*70)
    aci_predictor = ConformalPredictor(model, alpha=args.alpha, gamma=args.gamma)
    q_0 = aci_predictor.calibrate(X_cal, y_cal, binary=args.binary)
    print(f"Initial threshold (q_0): {q_0:.4f}")
    print(f"Target coverage: {1 - args.alpha:.2%}")

    # Evaluate on test set with adaptive updates
    print("\n" + "="*70)
    print("Step 3: Adaptive Conformal Prediction on Test Set")
    print("="*70)
    test_coverage, test_avg_size, final_q = aci_predictor.get_coverage_and_efficiency(
        X_test, y_test, binary=args.binary
    )
    print(f"Test set - Coverage: {test_coverage:.4f}, Avg set size: {test_avg_size:.4f}")
    print(f"Final threshold (q_T): {final_q:.4f}")
    print(f"Threshold drift: {final_q - q_0:+.4f}")

    print("\n" + "="*70)
    print("Step 4: Computing Final Metrics")
    print("="*70)

    with torch.no_grad():
        # Get standard predictions for all splits
        def get_preds(X, y):
            reward_pred = F.sigmoid(model(X).squeeze()) if args.binary else model(X).squeeze()
            reward_pred = reward_pred.detach().cpu().numpy()
            y_cpu = y.cpu().numpy()
            return reward_pred, y_cpu

        y_train_pred, y_train_cpu = get_preds(X_train, y_train)
        y_cal_pred, y_cal_cpu = get_preds(X_cal, y_cal)
        y_test_pred, y_test_cpu = get_preds(X_test, y_test)

    # Metrics
    metrics = {
        # Point prediction metrics
        "R2 on train": r2_score(y_train_cpu, y_train_pred),
        "R2 on cal": r2_score(y_cal_cpu, y_cal_pred),
        "R2 on test": r2_score(y_test_cpu, y_test_pred),
        "MAE on cal": mean_absolute_error(y_cal_cpu, y_cal_pred),
        "MAE on test": mean_absolute_error(y_test_cpu, y_test_pred),
        "MSE on cal": mean_squared_error(y_cal_cpu, y_cal_pred),
        "MSE on test": mean_squared_error(y_test_cpu, y_test_pred),
        "RMSE on cal": np.sqrt(mean_squared_error(y_cal_cpu, y_cal_pred)),
        "RMSE on test": np.sqrt(mean_squared_error(y_test_cpu, y_test_pred)),
        "AUROC on cal": roc_auc_score(y_cal_cpu, y_cal_pred),
        "AUROC on test": roc_auc_score(y_test_cpu, y_test_pred),

        # ACI metrics
        "CP alpha": args.alpha,
        "CP gamma": args.gamma,
        "CP q_0 (initial)": q_0,
        "CP q_T (final)": final_q,
        "CP threshold drift": final_q - q_0,
        "CP coverage on test": test_coverage,
        "CP interval size on test": test_avg_size,
    }
    
    metrics = refine_dict(metrics)
    print("\n--- Final Performance ---")
    for metric, value in metrics.items():
        if isinstance(value, float):
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")
    
    save_metrics(args, metrics)


if __name__ == '__main__':
    main()
