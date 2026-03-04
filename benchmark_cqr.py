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
    """
    Neural network for quantile regression.
    Predicts both lower and upper quantiles simultaneously.
    """
    def __init__(self, input_size, hidden_dim_str):
        super(Model, self).__init__()
        hidden_dims = [input_size] + list(map(int, hidden_dim_str.split(',')))
        self.layers = nn.ModuleList(
            nn.Linear(hidden_dims[i], hidden_dims[i + 1]) for i in range(len(hidden_dims) - 1)
        )
        # Separate output heads for lower and upper quantiles
        self.lower_quantile_head = nn.Linear(hidden_dims[-1], 1)
        self.upper_quantile_head = nn.Linear(hidden_dims[-1], 1)

    def forward(self, x):
        # Shared feature extraction
        for layer in self.layers:
            x = F.leaky_relu(layer(x))
        
        # Predict both quantiles
        q_low = self.lower_quantile_head(x)
        q_high = self.upper_quantile_head(x)
        
        return q_low, q_high


def quantile_loss(y_pred, y_true, quantile):
    """
    Quantile regression loss (also known as pinball loss).
    
    Args:
        y_pred: Predicted values
        y_true: True values
        quantile: Target quantile (e.g., 0.05 for lower, 0.95 for upper)
    
    Returns:
        Quantile loss
    """
    errors = y_true - y_pred
    loss = torch.max((quantile - 1) * errors, quantile * errors)
    return loss.mean()


class ConformalPredictor:
    """
    Conformalized Quantile Regression (CQR) for uncertainty quantification.
    
    This class implements CQR which combines quantile regression with conformal prediction
    to provide prediction intervals with guaranteed coverage.
    
    Reference:
    Romano, Y., Patterson, E., & Candes, E. (2019). 
    Conformalized quantile regression. NeurIPS 2019.
    """
    def __init__(self, model, alpha=0.1):
        """
        Args:
            model: The trained quantile regression model
            alpha: Miscoverage rate (default 0.1 for 90% coverage)
        """
        self.model = model
        self.alpha = alpha
        self.q_hat = None  # Conformal correction term
        
    def calibrate(self, X_cal, y_cal):
        """
        Calibrate the CQR predictor using a calibration dataset.
        
        Args:
            X_cal: Calibration features
            y_cal: Calibration labels
            
        Returns:
            q_hat: The conformal correction term
        """
        self.model.eval()
        with torch.no_grad():
            q_low, q_high = self.model(X_cal)
            q_low = q_low.squeeze()
            q_high = q_high.squeeze()
            
            # Compute nonconformity scores
            # Score = max(q_low - y, y - q_high)
            # This measures how much the true value is outside the predicted interval
            scores = torch.max(q_low - y_cal, y_cal - q_high)
        
        # Compute the (1-alpha) quantile of nonconformity scores
        n = len(scores)
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        self.q_hat = torch.quantile(scores, q_level).item()
        
        return self.q_hat
    
    def predict_with_uncertainty(self, X):
        """
        Make predictions with CQR prediction intervals.
        
        Args:
            X: Input features
            
        Returns:
            predictions: Point predictions (midpoint of interval)
            lower_bounds: Lower bounds of prediction intervals
            upper_bounds: Upper bounds of prediction intervals
            set_sizes: Sizes of prediction intervals
        """
        self.model.eval()
        with torch.no_grad():
            q_low, q_high = self.model(X)
            q_low = q_low.squeeze()
            q_high = q_high.squeeze()
            
            # Apply conformal correction
            # Expand the intervals by q_hat on both sides
            lower_bounds = q_low - self.q_hat
            upper_bounds = q_high + self.q_hat
            
            # Point prediction: midpoint of the interval
            predictions = (lower_bounds + upper_bounds) / 2
            
            # Interval size
            set_sizes = upper_bounds - lower_bounds
        
        return predictions, lower_bounds, upper_bounds, set_sizes
    
    def get_coverage_and_efficiency(self, X, y):
        """
        Evaluate coverage and efficiency of CQR prediction intervals.
        
        Args:
            X: Test features
            y: Test labels
            
        Returns:
            coverage: Empirical coverage rate
            avg_set_size: Average prediction interval size
        """
        predictions, lower_bounds, upper_bounds, set_sizes = self.predict_with_uncertainty(X)
        
        # Coverage: true value is in interval
        coverage = ((y >= lower_bounds) & (y <= upper_bounds)).float().mean().item()
        
        # Average interval size
        avg_set_size = set_sizes.mean().item()
        
        return coverage, avg_set_size


def parse_arguments():
    # Pre-parse only data_name to select dataset defaults
    pre_parser = ArgumentParser(add_help=False)
    pre_parser.add_argument("--data_name", type=str, default="hs")
    pre_args, _ = pre_parser.parse_known_args()

    # Base defaults
    base_defaults = {
        "desc": "cqr",
        "is_training": True,
        "output_dir": f"./results/cache/cqr/{pre_args.data_name}",
        "data_root": "./embeddings/clean",
        "model_name": "FsfairX-LLaMA3-RM-v0.1",
        "estimator_name": "cqr",
        "data_name": pre_args.data_name,
        "lr": 0.0005,
        "num_epochs": 600,
        "batch_size": 512,
        "hidden_dim": "256,64",
        "patience": 30,
        "seed": 42,
        "l2_reg": 1e-6,
        "w_reg": 1.0,  # Weight for regularization term in loss
        "rerun": False,
        "binary": False,  # CQR only supports regression (binary must be False)
        "use_tqdm": True,
        "alpha": 0.1,  # Miscoverage rate (90% coverage)
    }

    dataset_defaults = {
        "hs": {
            "l2_reg": 1e-6,
            "w_reg": 1.0,
            "alpha": 0.1,
        },
        "ufb": {
            "l2_reg": 1e-5,
            "w_reg": 1.0,
            "alpha": 0.1,
        },
        "saferlhf": {
            "l2_reg": 1e-7,
            "w_reg": 1.0,
            "alpha": 0.1,
        },
    }
    ds_defaults = dataset_defaults.get(pre_args.data_name, {})
    merged_defaults = {**base_defaults, **ds_defaults}

    # Full parser
    parser = ArgumentParser(description="Conformalized Quantile Regression for Reward Models")
    parser.add_argument("--desc", type=str)
    parser.add_argument("--is_training", type=str2bool)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--data_root", type=str)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--data_name", type=str)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--num_epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--hidden_dim", type=str, help="Hidden dimensions, e.g., '128,64'")
    parser.add_argument("--patience", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--l2_reg", type=float, help="L2 regularization coefficient")
    parser.add_argument("--w_reg", type=float, help="Weight for regularization term in loss")
    parser.add_argument("--rerun", type=str2bool, help="Whether to rerun the experiment")
    parser.add_argument("--binary", type=str2bool, help="Whether to use binary labels (must be False for CQR)")
    parser.add_argument("--use_tqdm", type=str2bool, help="Whether to use tqdm progress bar")
    parser.add_argument("--alpha", type=float, help="Miscoverage rate for CQR")

    parser.set_defaults(**merged_defaults)
    args = parser.parse_args()
    return args


def train(model, train_loader, optimizer, num_epochs, patience, alpha, args):
    """
    Train the quantile regression model.
    
    The model learns to predict both lower (alpha/2) and upper (1-alpha/2) quantiles.
    """
    if not args.is_training:
        return

    best_loss = float('inf')
    patience_counter = 0

    # Quantiles to predict
    q_low = alpha / 2
    q_high = 1 - alpha / 2

    for epoch in range(num_epochs):
        epoch_loss = 0
        model.train()

        bar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{num_epochs}", leave=False) if args.use_tqdm else train_loader
        for batch_X, batch_y in bar:
            optimizer.zero_grad()
            
            # Predict both quantiles
            pred_q_low, pred_q_high = model(batch_X)
            pred_q_low = pred_q_low.squeeze()
            pred_q_high = pred_q_high.squeeze()
            
            # Compute quantile losses
            loss_low = quantile_loss(pred_q_low, batch_y, q_low)
            loss_high = quantile_loss(pred_q_high, batch_y, q_high)
            loss = loss_low + loss_high
            
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
    
    # CQR only supports regression tasks
    assert not args.binary, "CQR (Conformalized Quantile Regression) only supports regression tasks. Please set --binary False."

    if args.is_training and os.path.exists(f"{args.output_dir}/performance.yaml") and not args.rerun:
        print(f"The path {args.output_dir}/performance.yaml exists!!")
        sys.exit()
    os.makedirs(args.output_dir, exist_ok=True)

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("="*70)
    print("Conformalized Quantile Regression (CQR) for Reward Modeling")
    print("="*70)
    print("Loading preprocessed data from Safetensors file...")
    
    # Load all data from preprocessed file
    data_file = f"{args.data_root}/{args.model_name}_{args.data_name}.safetensors"
    
    # Load continuous labels (CQR is designed for regression)
    X_train, y_train, X_cal, y_cal, X_test, y_test = load_data(
        data_file, device,
        keys=["X_train", "y_train", "X_cal", "y_cal", "X_test", "y_test"]
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Calibration set: {X_cal.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Target quantiles: {args.alpha/2:.3f} (lower) and {1-args.alpha/2:.3f} (upper)")

    # Train quantile regression model
    print("\n" + "="*70)
    print("Step 1: Training Conformalized Quantile Regression Model")
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
        alpha=args.alpha,
        args=args
    )
    
    # Load best model
    model.load_state_dict(torch.load(f'{args.output_dir}/best_model.pth'))
    model.eval()

    # CQR Calibration
    print("\n" + "="*70)
    print("Step 2: Calibrating CQR Predictor")
    print("="*70)
    cqr_predictor = ConformalPredictor(model, alpha=args.alpha)
    q_hat = cqr_predictor.calibrate(X_cal, y_cal)
    print(f"CQR correction term (q_hat): {q_hat:.4f}")
    print(f"Expected coverage: {1 - args.alpha:.2%}")

    # Evaluate on test set
    test_coverage, test_avg_size = cqr_predictor.get_coverage_and_efficiency(X_test, y_test)
    print(f"Test set - Coverage: {test_coverage:.4f}, Avg interval size: {test_avg_size:.4f}")

    print("\n" + "="*70)
    print("Step 3: Computing Final Metrics")
    print("="*70)

    with torch.no_grad():
        # Get CQR predictions for all splits
        def get_preds(X, y):
            preds, _, _, _ = cqr_predictor.predict_with_uncertainty(X)
            reward_pred = preds.cpu().numpy()
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
        "AUROC on cal": roc_auc_score((y_cal_cpu > np.median(y_cal_cpu)).astype(int), y_cal_pred),
        "AUROC on test": roc_auc_score((y_test_cpu > np.median(y_test_cpu)).astype(int), y_test_pred),

        # CQR metrics
        "CQR alpha": args.alpha,
        "CQR q_hat": q_hat,
        "CQR coverage on test": test_coverage,
        "CQR interval size on test": test_avg_size,
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
