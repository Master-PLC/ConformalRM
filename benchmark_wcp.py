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


def compute_importance_weights(weight_model, X_cal, device):
    """
    Compute importance weights for calibration samples.
    
    Args:
        weight_model: Trained weight estimation model
        X_cal: Calibration features
        device: Device to use
        
    Returns:
        weights: Importance weights for calibration samples
    """
    weight_model.eval()
    with torch.no_grad():
        logits = weight_model(X_cal).squeeze()
        probs = torch.sigmoid(logits)  # P(Y=1|X) = probability of being from test set
        
        # w(x) = p_test(x) / p_cal(x) = P(Y=1|X) / P(Y=0|X)
        weights = probs / (1 - probs + 1e-6)  # Add small epsilon to avoid division by zero
        
        # Clip extreme weights to improve stability
        weights = torch.clamp(weights, min=0.01, max=100.0)
        
    print(f"  Weight statistics: min={weights.min():.4f}, max={weights.max():.4f}, mean={weights.mean():.4f}")
    
    return weights


def weighted_quantile(values, weights, quantile_level):
    """
    Compute weighted quantile.
    
    Args:
        values: Values to compute quantile of
        weights: Importance weights for each value
        quantile_level: Target quantile level (e.g., 0.9 for 90th percentile)
        
    Returns:
        Weighted quantile value
    """
    # Sort values and corresponding weights
    sorted_indices = torch.argsort(values)
    sorted_values = values[sorted_indices]
    sorted_weights = weights[sorted_indices]
    
    # Compute cumulative weights
    cumsum_weights = torch.cumsum(sorted_weights, dim=0)
    total_weight = cumsum_weights[-1]
    
    # Normalize to [0, 1]
    cumsum_normalized = cumsum_weights / total_weight
    
    # Find the quantile
    # We want the smallest value where cumsum >= quantile_level
    mask = cumsum_normalized >= quantile_level
    if mask.any():
        quantile_idx = torch.argmax(mask.float())
        return sorted_values[quantile_idx].item()
    else:
        return sorted_values[-1].item()


class ConformalPredictor:
    """
    Weighted Conformal Prediction for handling covariate shift.
    
    This class implements weighted conformal prediction which adjusts for distribution shift
    between training and test sets using importance weights.
    
    Reference:
    Tibshirani, R. J., Foygel Barber, R., Candes, E., & Ramdas, A. (2019).
    Conformal prediction under covariate shift. NeurIPS 2019.
    """
    def __init__(self, model, alpha=0.1):
        """
        Args:
            model: The trained reward model
            alpha: Miscoverage rate (default 0.1 for 90% coverage)
        """
        self.model = model
        self.alpha = alpha
        self.q_hat = None  # Weighted conformal correction term
        
    def calibrate(self, X_cal, y_cal, weights, binary=True):
        """
        Calibrate using weighted conformal prediction.
        
        Args:
            X_cal: Calibration features
            y_cal: Calibration labels
            weights: Importance weights for calibration samples (if None, uses uniform weights)
            binary: Whether using binary classification
            
        Returns:
            q_hat: The weighted conformal correction term
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
        
        # Weighted conformal prediction
        # Adjust quantile level for weighted case
        n = len(scores)
        adjusted_level = (1 - self.alpha) * (1 + 1/n)
        self.q_hat = weighted_quantile(scores, weights, adjusted_level)
        
        return self.q_hat
    
    def predict_with_uncertainty(self, X, binary=True):
        """
        Make predictions with weighted conformal prediction intervals.
        
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
                # A label is included if 1 - P(label) <= q_hat
                # Equivalently, P(label) >= 1 - q_hat
                threshold = 1 - self.q_hat
                
                # Check which labels are in the prediction set
                pred_set_positive = (probs >= threshold).float()
                pred_set_negative = ((1 - probs) >= threshold).float()
                
                # Set sizes: number of labels in prediction set
                set_sizes = pred_set_positive + pred_set_negative
                
                # For binary, we represent uncertainty by the prediction set
                # Lower and upper bounds represent the range
                lower_bounds = torch.where(pred_set_negative > 0,
                                          torch.zeros_like(probs),
                                          probs - self.q_hat)
                upper_bounds = torch.where(pred_set_positive > 0,
                                          torch.ones_like(probs),
                                          probs + self.q_hat)
            else:
                # For regression
                predictions = outputs
                lower_bounds = outputs - self.q_hat
                upper_bounds = outputs + self.q_hat
                set_sizes = (upper_bounds - lower_bounds).abs()
        
        return predictions, lower_bounds, upper_bounds, set_sizes
    
    def get_coverage_and_efficiency(self, X, y, binary=True):
        """
        Evaluate coverage and efficiency of weighted conformal prediction intervals.
        
        Args:
            X: Test features
            y: Test labels
            binary: Whether using binary classification
            
        Returns:
            coverage: Empirical coverage rate
            avg_set_size: Average prediction set size
        """
        predictions, lower_bounds, upper_bounds, set_sizes = self.predict_with_uncertainty(X, binary)
        
        if binary:
            # Coverage: true label is in prediction set
            y_binary = (y > 0.5).float()
            # For binary classification, check if true label has sufficient probability
            probs = predictions
            threshold = 1 - self.q_hat
            
            coverage_mask = torch.where(
                y_binary > 0.5,
                probs >= threshold,      # Positive label covered
                (1 - probs) >= threshold # Negative label covered
            )
            coverage = coverage_mask.float().mean().item()
        else:
            # For regression, check if true value is in interval
            coverage = ((y >= lower_bounds) & (y <= upper_bounds)).float().mean().item()
        
        avg_set_size = set_sizes.mean().item()
        
        return coverage, avg_set_size


def parse_arguments():
    # Pre-parse only data_name to select dataset defaults
    pre_parser = ArgumentParser(add_help=False)
    pre_parser.add_argument("--data_name", type=str, default="hs")
    pre_args, _ = pre_parser.parse_known_args()

    # Base defaults
    base_defaults = {
        "desc": "wcp",
        "is_training": True,
        "output_dir": f"./results/cache/wcp/{pre_args.data_name}",
        "data_root": "./embeddings/clean",
        "model_name": "FsfairX-LLaMA3-RM-v0.1",
        "estimator_name": "wcp",
        "data_name": pre_args.data_name,
        "lr": 0.0005,
        "num_epochs": 600,
        "batch_size": 512,
        "hidden_dim": "256,64",
        "weight_hidden_dim": "128,64",
        "patience": 30,
        "seed": 42,
        "l2_reg": 1e-6,
        "w_reg": 1.0,
        "l2_weight": 1e-6,
        "w_weight": 1.0,
        "rerun": False,
        "binary": True,  # Support both binary and regression
        "use_tqdm": True,
        "alpha": 0.1,
    }

    dataset_defaults = {
        "hs": {
            "l2_reg": 1e-6,
            "w_reg": 1.0,
            "l2_weight": 1e-6,
            "w_weight": 1.0,
            "alpha": 0.1,
        },
        "ufb": {
            "l2_reg": 1e-5,
            "w_reg": 1.0,
            "l2_weight": 1e-5,
            "w_weight": 1.0,
            "alpha": 0.1,
        },
        "saferlhf": {
            "l2_reg": 1e-7,
            "w_reg": 1.0,
            "l2_weight": 1e-7,
            "w_weight": 1.0,
            "alpha": 0.1,
        },
    }
    ds_defaults = dataset_defaults.get(pre_args.data_name, {})
    merged_defaults = {**base_defaults, **ds_defaults}

    # Full parser
    parser = ArgumentParser(description="Weighted Conformal Prediction for Reward Models")
    parser.add_argument("--desc", type=str)
    parser.add_argument("--is_training", type=str2bool)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--data_root", type=str)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--data_name", type=str)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--num_epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--hidden_dim", type=str, help="Hidden dimensions for quantile model")
    parser.add_argument("--weight_hidden_dim", type=str, help="Hidden dimensions for weight estimator")
    parser.add_argument("--patience", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--l2_reg", type=float)
    parser.add_argument("--w_reg", type=float)
    parser.add_argument("--l2_weight", type=float)
    parser.add_argument("--w_weight", type=float)
    parser.add_argument("--rerun", type=str2bool)
    parser.add_argument("--binary", type=str2bool)
    parser.add_argument("--use_tqdm", type=str2bool)
    parser.add_argument("--alpha", type=float)

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


def train_weight_estimator(weight_model, weight_loader, optimizer, num_epochs, patience, args):
    """
    Train the weight estimator using binary classification.
    
    Args:
        weight_model: The weight estimation model
        weight_loader: DataLoader for weight estimation
        optimizer: Optimizer for weight model
        num_epochs: Number of training epochs
        args: Arguments
    """
    if not args.is_training:
        return

    best_loss = float('inf')
    patience_counter = 0

    criterion = nn.BCEWithLogitsLoss()
    
    weight_model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch_X, batch_y in weight_loader:
            optimizer.zero_grad()
            logits = weight_model(batch_X).squeeze()
            loss = criterion(logits, batch_y)
            
            weighted_loss = args.w_weight * loss
            weighted_loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        monitor_loss = epoch_loss / len(weight_loader)
        if (epoch + 1) % 10 == 0:
            print(f"  Weight estimation epoch {epoch+1}/{num_epochs}, Loss: {monitor_loss:.4f}")
        
        if monitor_loss < best_loss:
            best_loss = monitor_loss
            patience_counter = 0
            torch.save(weight_model.state_dict(), f'{args.output_dir}/best_weight_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping for weight estimator after {epoch + 1} epochs.")
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
    print("Weighted Conformal Prediction (WCP) for Reward Modeling")
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
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_w)

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

    # Estimate importance weights
    print("\n" + "="*70)
    print("Step 2: Estimating Importance Weights")
    print("="*70)
    print("Training classifier to distinguish calibration from test distribution...")
    
    # Create binary classification dataset
    # Calibration samples: label = 0, Test samples: label = 1
    X_combined = torch.cat([X_cal, X_test], dim=0)
    y_combined = torch.cat([
        torch.zeros(X_cal.shape[0], device=device),
        torch.ones(X_test.shape[0], device=device)
    ], dim=0)
    
    # Create dataloader for weight estimation
    weight_loader = DataLoader(TensorDataset(X_combined, y_combined), batch_size=args.batch_size, shuffle=True)
    
    # Initialize weight estimator
    weight_model = Model(X_cal.shape[1], args.weight_hidden_dim).to(device)
    weight_optimizer = torch.optim.Adam(weight_model.parameters(), lr=args.lr, weight_decay=args.l2_weight)
    
    # Train weight estimator
    train_weight_estimator(
        weight_model=weight_model,
        weight_loader=weight_loader,
        optimizer=weight_optimizer,
        num_epochs=args.num_epochs,
        patience=args.patience,
        args=args
    )
    
    # Compute importance weights for calibration set
    cal_weights = compute_importance_weights(weight_model, X_cal, device)

    # Weighted Conformal Calibration
    print("\n" + "="*70)
    print("Step 3: Weighted Conformal Calibration")
    print("="*70)
    wcp_predictor = ConformalPredictor(model, alpha=args.alpha)
    q_hat = wcp_predictor.calibrate(X_cal, y_cal, weights=cal_weights, binary=args.binary)
    print(f"Weighted conformal correction term (q_hat): {q_hat:.4f}")
    print(f"Expected coverage: {1 - args.alpha:.2%}")

    # Evaluate on test set
    test_coverage, test_avg_size = wcp_predictor.get_coverage_and_efficiency(X_test, y_test, binary=args.binary)
    print(f"Test set - Coverage: {test_coverage:.4f}, Avg interval size: {test_avg_size:.4f}")

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

        # WCP metrics
        "CP alpha": args.alpha,
        "CP q_hat": q_hat,
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
