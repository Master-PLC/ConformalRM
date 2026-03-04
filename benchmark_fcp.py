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

# Try to import auto_LiRPA for precise bound estimation
try:
    from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
    LIRPA_AVAILABLE = True
except ImportError:
    LIRPA_AVAILABLE = False
    print("Warning: auto_LiRPA not available. Will use Monte Carlo sampling for bound estimation.")
    print("To install: git clone https://github.com/Verified-Intelligence/auto_LiRPA && cd auto_LiRPA && pip install -e .")


class Model(nn.Module):
    """Standard neural network for reward prediction."""
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


def compute_surrogate_feature(model, X, Y, num_steps=100, lr=0.1, binary=True):
    """
    Compute surrogate feature v such that model(v) = Y.
    
    In this setting, X is already the feature embedding from reward model.
    We find v in the same feature space such that model(v) = Y.
    
    This implements Algorithm 2 from the FeatureCP paper (Teng et al., 2023).
    """
    model.eval()
    
    # Initialize from input features (X is already the feature)
    v = X.clone()
    
    # Make v trainable
    v.requires_grad_(True)
    
    # Gradient descent to find surrogate feature
    optimizer = torch.optim.Adam([v], lr=lr)
    
    for step in range(num_steps):
        optimizer.zero_grad()
        
        # Predict from current surrogate features
        pred = model(v).squeeze()
        
        # Loss: make model(v) = Y
        if binary:
            loss = F.binary_cross_entropy_with_logits(pred, Y)
        else:
            loss = F.mse_loss(pred, Y)
        
        loss.backward()
        optimizer.step()
    
    return v.detach()


class ConformalPredictor:
    """
    Feature Conformal Prediction for uncertainty quantification.
    
    In this setting, input X is already the feature embedding from reward model.
    Feature CP finds surrogate features v in the same space such that model(v) = Y,
    and uses ||v - X|| as the nonconformity score.
    
    Reference:
    Teng, J., Wen, C., Zhang, D., Bengio, Y., Gao, Y., & Yuan, Y. (2023).
    Predictive Inference with Feature Conformal Prediction. ICLR 2023.
    """
    def __init__(self, model, alpha=0.1, surrogate_steps=100, surrogate_lr=0.1):
        self.model = model
        self.alpha = alpha
        self.surrogate_steps = surrogate_steps
        self.surrogate_lr = surrogate_lr
        self.q_hat = None
        
    def calibrate(self, X_cal, y_cal, binary=True):
        """
        Calibrate using calibration set.
        
        X_cal is already the feature embedding. We compute surrogate features v
        such that model(v) = y_cal, then use ||v - X_cal|| as nonconformity scores.
        """
        self.model.eval()
        
        # X_cal is already the feature embedding
        # Compute surrogate features v such that model(v) = Y
        surrogate_features = compute_surrogate_feature(
            self.model, X_cal, y_cal, 
            num_steps=self.surrogate_steps,
            lr=self.surrogate_lr,
            binary=binary
        )
        
        # Nonconformity score: ||v - X|| in feature space
        scores = torch.norm(surrogate_features - X_cal, dim=1)
        
        # Compute the (1-alpha) quantile
        n = len(scores)
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        self.q_hat = torch.quantile(scores, q_level).item()
        
        return self.q_hat
    
    def predict_with_uncertainty(self, X, binary=True, num_samples=1000,
                                use_lirpa=True, lirpa_method='IBP'):
        """
        Make predictions with feature conformal prediction intervals.
        
        X is already the feature embedding. We estimate bounds using either:
        1. LiRPA (auto_LiRPA): Precise bound estimation (if available)
        2. Monte Carlo: Sampling-based estimation (fallback)
        
        Args:
            X: Input features
            binary: Whether using binary classification
            num_samples: Number of MC samples (only used if LiRPA unavailable)
            use_lirpa: Whether to use LiRPA for precise bounds
            lirpa_method: LiRPA method ('IBP', 'CROWN', 'CROWN-Optimized')
        """
        self.model.eval()
        
        # Point predictions
        with torch.no_grad():
            outputs = self.model(X).squeeze()
            if binary:
                predictions = torch.sigmoid(outputs)
            else:
                predictions = outputs
        
        # Choose bound estimation method
        if use_lirpa:
            # Method 1: Interval Bound Propagation (certified bounds)
            lower_bounds, upper_bounds = self._compute_bounds_ibp(X, binary, lirpa_method)
        else:
            # Method 2: Monte Carlo sampling (approximate bounds)
            lower_bounds, upper_bounds = self._compute_bounds_mc(
                X, binary, num_samples
            )
        
        set_sizes = (upper_bounds - lower_bounds).abs()
        return predictions, lower_bounds, upper_bounds, set_sizes
    
    def _compute_bounds_lirpa(self, X, binary, method='CROWN-Optimized'):
        """
        Compute precise bounds using auto_LiRPA.
        
        This implements the Band Estimation method from the FeatureCP paper.
        """
        batch_size = X.shape[0]
        
        # Wrap model with BoundedModule
        # X is the input feature, model is the prediction head
        lirpa_model = BoundedModule(self.model, torch.empty_like(X))
        
        # Define perturbation: ||v - X|| <= q_hat (L2 norm)
        ptb = PerturbationLpNorm(norm=2, eps=self.q_hat)
        bounded_input = BoundedTensor(X, ptb)
        
        # Set optimization parameters for CROWN-Optimized
        if 'Optimized' in method:
            lirpa_model.set_bound_opts({
                'optimize_bound_args': {
                    'ob_iteration': 20,
                    'ob_lr': 0.1,
                    'ob_verbose': 0
                }
            })
        
        # Compute bounds
        lb, ub = lirpa_model.compute_bounds(x=(bounded_input,), method=method)
        
        # Apply sigmoid for binary classification
        if binary:
            lb = torch.sigmoid(lb)
            ub = torch.sigmoid(ub)
        
        return lb.squeeze(), ub.squeeze()
    
    def _compute_bounds_mc(self, X, binary, num_samples):
        """
        Compute bounds using Monte Carlo sampling (fallback method).
        """
        with torch.no_grad():
            batch_size, feature_dim = X.shape[0], X.shape[1]
            
            # Sample random directions
            random_dirs = torch.randn(batch_size, num_samples, feature_dim, device=X.device)
            random_dirs = random_dirs / torch.norm(random_dirs, dim=2, keepdim=True)
            
            # Sample random radii uniformly in [0, q_hat]
            random_radii = torch.rand(batch_size, num_samples, 1, device=X.device) * self.q_hat
            
            # Generate sampled features around X
            sampled_features = X.unsqueeze(1) + random_dirs * random_radii
            sampled_features = sampled_features.reshape(-1, feature_dim)
            
            # Predict from sampled features
            sampled_outputs = self.model(sampled_features).squeeze()
            sampled_outputs = sampled_outputs.reshape(batch_size, num_samples)
            
            if binary:
                sampled_preds = torch.sigmoid(sampled_outputs)
                lower_bounds = sampled_preds.min(dim=1)[0]
                upper_bounds = sampled_preds.max(dim=1)[0]
            else:
                lower_bounds = sampled_outputs.min(dim=1)[0]
                upper_bounds = sampled_outputs.max(dim=1)[0]
        
        return lower_bounds, upper_bounds
    
    def get_coverage_and_efficiency(self, X, y, binary=True, num_samples=1000,
                                    use_lirpa=True, lirpa_method='IBP'):
        """Evaluate coverage and efficiency."""
        predictions, lower_bounds, upper_bounds, set_sizes = self.predict_with_uncertainty(
            X, binary, num_samples, use_lirpa, lirpa_method
        )
        
        if binary:
            # For binary, check if true label is in prediction set
            y_probs = y  # Already in [0, 1] for binary
            coverage = ((y_probs >= lower_bounds) & (y_probs <= upper_bounds)).float().mean().item()
        else:
            # For regression, check if true value is in interval
            coverage = ((y >= lower_bounds) & (y <= upper_bounds)).float().mean().item()
        
        avg_set_size = set_sizes.mean().item()
        
        return coverage, avg_set_size


def parse_arguments():
    pre_parser = ArgumentParser(add_help=False)
    pre_parser.add_argument("--data_name", type=str, default="hs")
    pre_args, _ = pre_parser.parse_known_args()

    base_defaults = {
        "desc": "fcp",
        "is_training": True,
        "output_dir": f"./results/cache/fcp/{pre_args.data_name}",
        "data_root": "./embeddings/clean",
        "model_name": "FsfairX-LLaMA3-RM-v0.1",
        "estimator_name": "fcp",
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
        "binary": True,
        "use_tqdm": True,
        "alpha": 0.1,
        "surrogate_steps": 100,
        "surrogate_lr": 0.1,
        "num_samples": 1000,
        "use_lirpa": True,
        "lirpa_method": "IBP",
    }

    dataset_defaults = {
        "hs": {
            "l2_reg": 1e-6,
            "w_reg": 1.0,
            "alpha": 0.1,
            "surrogate_steps": 100,
            "surrogate_lr": 0.1,
        },
        "ufb": {
            "l2_reg": 1e-5,
            "w_reg": 1.0,
            "alpha": 0.1,
            "surrogate_steps": 100,
            "surrogate_lr": 0.1,
        },
        "saferlhf": {
            "l2_reg": 1e-7,
            "w_reg": 1.0,
            "alpha": 0.1,
            "surrogate_steps": 100,
            "surrogate_lr": 0.1,
        },
    }
    ds_defaults = dataset_defaults.get(pre_args.data_name, {})
    merged_defaults = {**base_defaults, **ds_defaults}

    parser = ArgumentParser(description="Feature Conformal Prediction for Reward Models")
    parser.add_argument("--desc", type=str)
    parser.add_argument("--is_training", type=str2bool)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--data_root", type=str)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--data_name", type=str)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--num_epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--hidden_dim", type=str)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--l2_reg", type=float)
    parser.add_argument("--w_reg", type=float)
    parser.add_argument("--rerun", type=str2bool)
    parser.add_argument("--binary", type=str2bool)
    parser.add_argument("--use_tqdm", type=str2bool)
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--surrogate_steps", type=int)
    parser.add_argument("--surrogate_lr", type=float)
    parser.add_argument("--num_samples", type=int)
    parser.add_argument("--use_lirpa", type=str2bool, help="Use auto_LiRPA for precise bounds")
    parser.add_argument("--lirpa_method", type=str, help="LiRPA method: IBP, IBP+backward, backward, CROWN-Optimized")

    parser.set_defaults(**merged_defaults)
    args = parser.parse_args()
    return args


def train(model, train_loader, optimizer, num_epochs, patience, args):
    """Train the reward model."""
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
    print("Feature Conformal Prediction (FCP) for Reward Modeling")
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
    print(f"Input feature dimension: {X_train.shape[1]}")
    print(f"Surrogate optimization: {args.surrogate_steps} steps with lr={args.surrogate_lr}")
    
    # Check bound estimation method
    if args.use_lirpa and LIRPA_AVAILABLE:
        print(f"Bound estimation: LiRPA ({args.lirpa_method})")
    else:
        if args.use_lirpa and not LIRPA_AVAILABLE:
            print("Warning: LiRPA requested but not available. Falling back to Monte Carlo.")
        print(f"Bound estimation: Monte Carlo (num_samples={args.num_samples})")

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

    # Feature Conformal Prediction Calibration
    print("\n" + "="*70)
    print("Step 2: Feature Conformal Calibration")
    print("="*70)
    print("Computing surrogate features on calibration set...")
    
    fcp_predictor = ConformalPredictor(
        model,
        alpha=args.alpha,
        surrogate_steps=args.surrogate_steps,
        surrogate_lr=args.surrogate_lr
    )
    q_hat = fcp_predictor.calibrate(X_cal, y_cal, binary=args.binary)
    print(f"Feature space quantile (q_hat): {q_hat:.4f}")
    print(f"Expected coverage: {1 - args.alpha:.2%}")

    # Evaluate on test set
    print("\n" + "="*70)
    print("Step 3: Evaluating Feature Conformal Prediction on Test Set")
    print("="*70)
    test_coverage, test_avg_size = fcp_predictor.get_coverage_and_efficiency(
        X_test, y_test, binary=args.binary, num_samples=args.num_samples,
        use_lirpa=args.use_lirpa, lirpa_method=args.lirpa_method
    )
    print(f"Test set - Coverage: {test_coverage:.4f}, Avg set size: {test_avg_size:.4f}")

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

        # FCP metrics
        "CP alpha": args.alpha,
        "CP q_hat": q_hat,
        "CP coverage on test": test_coverage,
        "CP interval size on test": test_avg_size,
        "CP surrogate steps": args.surrogate_steps,
        "CP surrogate lr": args.surrogate_lr,
        "CP bound method": "LiRPA" if (args.use_lirpa and LIRPA_AVAILABLE) else "MC",
        "CP lirpa method": args.lirpa_method if (args.use_lirpa and LIRPA_AVAILABLE) else "N/A",
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
