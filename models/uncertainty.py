"""
Complete Uncertainty Quantification System
Implements state-of-the-art uncertainty estimation

Types of Uncertainty:
1. Epistemic (Model) Uncertainty - Reducible with more data
2. Aleatoric (Data) Uncertainty - Irreducible noise in data
3. Distributional Uncertainty - Out-of-distribution detection
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, NamedTuple
from dataclasses import dataclass
from scipy import stats
from scipy.special import softmax
import warnings

from utils.logger import log


@dataclass
class UncertaintyEstimate:
    """Complete uncertainty estimate for a prediction"""
    # Probabilities
    mean_probs: np.ndarray           # Mean predicted probabilities
    prob_std: np.ndarray             # Std of predicted probabilities
    
    # Class prediction
    predicted_class: int
    prediction_confidence: float     # Confidence in prediction
    
    # Uncertainty components
    epistemic: float                 # Model uncertainty (0-1)
    aleatoric: float                 # Data uncertainty (0-1)
    total: float                     # Total uncertainty (0-1)
    
    # Calibrated confidence
    calibrated_confidence: float     # After temperature scaling
    
    # Distribution info
    entropy: float                   # Predictive entropy
    mutual_info: float               # Epistemic uncertainty (MI)
    
    # Reliability
    is_reliable: bool                # Whether prediction is reliable
    reliability_score: float         # Overall reliability (0-1)
    
    # Prediction interval
    lower_bound: float               # Lower bound of prediction
    upper_bound: float               # Upper bound of prediction
    
    # Out-of-distribution
    ood_score: float                 # How OOD is this input (0-1)
    is_ood: bool                     # Is this out-of-distribution?
    
    def to_dict(self) -> Dict:
        return {
            'predicted_class': self.predicted_class,
            'confidence': self.prediction_confidence,
            'calibrated_confidence': self.calibrated_confidence,
            'epistemic_uncertainty': self.epistemic,
            'aleatoric_uncertainty': self.aleatoric,
            'total_uncertainty': self.total,
            'is_reliable': self.is_reliable,
            'reliability_score': self.reliability_score,
            'is_ood': self.is_ood
        }


class MonteCarloDropout:
    """
    Monte Carlo Dropout for epistemic uncertainty estimation
    
    During inference, we keep dropout enabled and run multiple forward passes.
    The variance in predictions indicates model uncertainty.
    """
    
    def __init__(self, model: nn.Module, num_samples: int = 30):
        self.model = model
        self.num_samples = num_samples
    
    def predict(self, x: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run MC Dropout inference
        
        Returns:
            mean_probs: Mean probabilities across samples
            std_probs: Standard deviation of probabilities
        """
        self.model.train()  # Enable dropout
        
        samples = []
        with torch.no_grad():
            for _ in range(self.num_samples):
                logits, _ = self.model(x)
                probs = F.softmax(logits, dim=-1)
                samples.append(probs.cpu().numpy())
        
        self.model.eval()  # Restore eval mode
        
        samples = np.array(samples)  # (num_samples, batch, num_classes)
        mean_probs = samples.mean(axis=0)
        std_probs = samples.std(axis=0)
        
        return mean_probs, std_probs


class DeepEnsembleUncertainty:
    """
    Deep Ensemble uncertainty using multiple models
    
    Different random initializations lead to different local minima,
    providing a measure of epistemic uncertainty.
    """
    
    def __init__(self, models: List[nn.Module]):
        self.models = models
    
    def predict(self, x: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Run ensemble prediction
        
        Returns:
            mean_probs: Mean probabilities
            std_probs: Std of probabilities (epistemic uncertainty)
            disagreement: Model disagreement score
        """
        predictions = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                logits, _ = model(x)
                probs = F.softmax(logits, dim=-1)
                predictions.append(probs.cpu().numpy())
        
        predictions = np.array(predictions)  # (num_models, batch, num_classes)
        mean_probs = predictions.mean(axis=0)
        std_probs = predictions.std(axis=0)
        
        # Calculate disagreement (how much models disagree)
        pred_classes = predictions.argmax(axis=-1)  # (num_models, batch)
        disagreement = 1 - np.apply_along_axis(
            lambda x: np.bincount(x, minlength=predictions.shape[-1]).max() / len(x),
            0, pred_classes
        )
        
        return mean_probs, std_probs, float(disagreement.mean())


class EvidentialUncertainty(nn.Module):
    """
    Evidential Deep Learning for uncertainty
    
    Instead of predicting probabilities, predicts parameters of a 
    Dirichlet distribution, providing principled uncertainty estimates.
    """
    
    def __init__(self, input_dim: int, num_classes: int = 3):
        super().__init__()
        self.num_classes = num_classes
        
        # Output layer predicts evidence (positive values)
        self.evidence_layer = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, num_classes),
            nn.Softplus()  # Ensure positive evidence
        )
    
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Returns:
            probs: Predicted probabilities
            uncertainty: Total uncertainty
            evidence: Evidence values
        """
        # Evidence (alpha - 1 for Dirichlet)
        evidence = self.evidence_layer(features)
        
        # Dirichlet parameters
        alpha = evidence + 1
        
        # Dirichlet strength
        S = alpha.sum(dim=-1, keepdim=True)
        
        # Expected probabilities
        probs = alpha / S
        
        # Uncertainty (vacuity)
        uncertainty = self.num_classes / S
        
        return probs.squeeze(-1), uncertainty.squeeze(-1), evidence
    
    def compute_loss(self, evidence: torch.Tensor, targets: torch.Tensor, 
                     epoch: int, total_epochs: int) -> torch.Tensor:
        """
        Evidential loss with KL divergence regularization
        """
        alpha = evidence + 1
        S = alpha.sum(dim=-1, keepdim=True)
        
        # Type II Maximum Likelihood
        A = (targets - alpha / S) ** 2
        B = alpha * (S - alpha) / (S ** 2 * (S + 1))
        loss_mse = (A + B).sum(dim=-1).mean()
        
        # KL divergence regularization (annealed)
        annealing = min(1.0, epoch / (total_epochs * 0.5))
        
        alpha_tilde = targets + (1 - targets) * alpha
        S_tilde = alpha_tilde.sum(dim=-1, keepdim=True)
        
        kl = torch.lgamma(S_tilde) - torch.lgamma(torch.tensor(self.num_classes, dtype=torch.float))
        kl -= torch.lgamma(alpha_tilde).sum(dim=-1, keepdim=True)
        kl += ((alpha_tilde - 1) * (torch.digamma(alpha_tilde) - torch.digamma(S_tilde))).sum(dim=-1, keepdim=True)
        
        loss_kl = annealing * kl.mean()
        
        return loss_mse + loss_kl


class TemperatureScaling:
    """
    Temperature scaling for confidence calibration
    
    Learns a single temperature parameter to calibrate confidence.
    Simple but effective post-hoc calibration method.
    """
    
    def __init__(self):
        self.temperature = 1.0
        self.fitted = False
    
    def fit(self, logits: np.ndarray, labels: np.ndarray) -> float:
        """
        Fit temperature on validation set
        
        Args:
            logits: Model logits (N, num_classes)
            labels: True labels (N,)
            
        Returns:
            Fitted temperature
        """
        from scipy.optimize import minimize_scalar
        
        def nll(T):
            scaled = logits / T
            probs = softmax(scaled, axis=1)
            log_probs = np.log(probs[np.arange(len(labels)), labels] + 1e-10)
            return -log_probs.mean()
        
        result = minimize_scalar(nll, bounds=(0.1, 10.0), method='bounded')
        self.temperature = result.x
        self.fitted = True
        
        log.info(f"Temperature scaling fitted: T={self.temperature:.4f}")
        return self.temperature
    
    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        """Apply temperature scaling to logits"""
        if not self.fitted:
            return logits
        return logits / self.temperature
    
    def calibrate_probs(self, probs: np.ndarray) -> np.ndarray:
        """Apply temperature scaling to probabilities"""
        if not self.fitted:
            return probs
        
        # Convert to logits, scale, convert back
        eps = 1e-10
        logits = np.log(probs + eps)
        scaled = logits / self.temperature
        return softmax(scaled, axis=-1)


class IsotonicCalibration:
    """
    Isotonic regression for probability calibration
    Non-parametric, more flexible than temperature scaling
    """
    
    def __init__(self):
        from sklearn.isotonic import IsotonicRegression
        self.calibrators = {}
        self.fitted = False
    
    def fit(self, probs: np.ndarray, labels: np.ndarray):
        """Fit isotonic calibrators for each class"""
        from sklearn.isotonic import IsotonicRegression
        
        num_classes = probs.shape[1]
        
        for c in range(num_classes):
            # Binary: is this class correct?
            binary_labels = (labels == c).astype(float)
            
            ir = IsotonicRegression(out_of_bounds='clip')
            ir.fit(probs[:, c], binary_labels)
            self.calibrators[c] = ir
        
        self.fitted = True
        log.info("Isotonic calibration fitted")
    
    def calibrate(self, probs: np.ndarray) -> np.ndarray:
        """Apply isotonic calibration"""
        if not self.fitted:
            return probs
        
        calibrated = np.zeros_like(probs)
        for c, ir in self.calibrators.items():
            calibrated[:, c] = ir.predict(probs[:, c])
        
        # Normalize to sum to 1
        calibrated = calibrated / (calibrated.sum(axis=1, keepdims=True) + 1e-10)
        return calibrated


class OutOfDistributionDetector:
    """
    Detects out-of-distribution inputs
    
    Methods:
    1. Maximum softmax probability
    2. Entropy-based
    3. Energy-based (more robust)
    4. Mahalanobis distance
    """
    
    def __init__(self, method: str = 'energy'):
        self.method = method
        self.threshold = None
        
        # For Mahalanobis
        self.mean = None
        self.precision = None
    
    def fit(self, features: np.ndarray, percentile: float = 95):
        """
        Fit OOD detector on in-distribution data
        
        Args:
            features: Feature representations from training data
            percentile: Percentile for threshold
        """
        if self.method == 'mahalanobis':
            self.mean = features.mean(axis=0)
            cov = np.cov(features.T) + np.eye(features.shape[1]) * 1e-6
            self.precision = np.linalg.inv(cov)
        
        # Compute scores on training data
        scores = self._compute_scores(features)
        self.threshold = np.percentile(scores, percentile)
        
        log.info(f"OOD detector fitted with threshold: {self.threshold:.4f}")
    
    def _compute_scores(self, features: np.ndarray, logits: np.ndarray = None) -> np.ndarray:
        """Compute OOD scores (higher = more OOD)"""
        if self.method == 'msp':
            # Maximum Softmax Probability (lower = more OOD)
            probs = softmax(logits, axis=1)
            return -probs.max(axis=1)
        
        elif self.method == 'entropy':
            probs = softmax(logits, axis=1)
            entropy = -np.sum(probs * np.log(probs + 1e-10), axis=1)
            return entropy
        
        elif self.method == 'energy':
            # Energy-based OOD (Grathwohl et al.)
            T = 1.0
            energy = -T * np.log(np.sum(np.exp(logits / T), axis=1) + 1e-10)
            return -energy  # Higher energy = more in-distribution
        
        elif self.method == 'mahalanobis':
            diff = features - self.mean
            scores = np.sqrt(np.sum(diff @ self.precision * diff, axis=1))
            return scores
        
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def detect(self, features: np.ndarray, logits: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect OOD samples
        
        Returns:
            scores: OOD scores (higher = more OOD)
            is_ood: Boolean mask of OOD samples
        """
        scores = self._compute_scores(features, logits)
        is_ood = scores > self.threshold if self.threshold else np.zeros(len(scores), dtype=bool)
        
        # Normalize scores to 0-1
        if self.threshold:
            normalized = scores / (2 * self.threshold)
            normalized = np.clip(normalized, 0, 1)
        else:
            normalized = np.zeros_like(scores)
        
        return normalized, is_ood


class UncertaintyQuantifier:
    """
    Main class for complete uncertainty quantification
    Combines all uncertainty estimation methods
    """
    
    def __init__(
        self,
        models: List[nn.Module],
        device: str = 'cuda',
        mc_samples: int = 30,
        calibration_method: str = 'temperature'  # or 'isotonic'
    ):
        self.models = models
        self.device = device
        self.mc_samples = mc_samples
        
        # Uncertainty estimators
        self.mc_dropout = MonteCarloDropout(models[0], mc_samples)
        self.ensemble = DeepEnsembleUncertainty(models)
        
        # Calibration
        if calibration_method == 'temperature':
            self.calibrator = TemperatureScaling()
        else:
            self.calibrator = IsotonicCalibration()
        
        # OOD detection
        self.ood_detector = OutOfDistributionDetector(method='energy')
        
        # Thresholds
        self.reliability_threshold = 0.6
        self.ood_threshold = 0.7
    
    def fit_calibration(self, val_logits: np.ndarray, val_labels: np.ndarray):
        """Fit calibration on validation set"""
        if isinstance(self.calibrator, TemperatureScaling):
            self.calibrator.fit(val_logits, val_labels)
        else:
            val_probs = softmax(val_logits, axis=1)
            self.calibrator.fit(val_probs, val_labels)
    
    def fit_ood(self, train_features: np.ndarray):
        """Fit OOD detector on training features"""
        self.ood_detector.fit(train_features)
    
    def quantify(
        self, 
        x: torch.Tensor,
        features: np.ndarray = None
    ) -> UncertaintyEstimate:
        """
        Complete uncertainty quantification
        
        Args:
            x: Input tensor
            features: Optional feature representation for OOD detection
            
        Returns:
            UncertaintyEstimate with all uncertainty metrics
        """
        x = x.to(self.device)
        
        # 1. MC Dropout uncertainty
        mc_mean, mc_std = self.mc_dropout.predict(x)
        mc_mean = mc_mean[0]  # Remove batch dim
        mc_std = mc_std[0]
        
        # 2. Ensemble uncertainty
        ens_mean, ens_std, disagreement = self.ensemble.predict(x)
        ens_mean = ens_mean[0]
        ens_std = ens_std[0]
        
        # 3. Combine predictions
        # Weight MC and ensemble equally
        mean_probs = (mc_mean + ens_mean) / 2
        prob_std = (mc_std + ens_std) / 2
        
        # 4. Epistemic uncertainty (model uncertainty)
        # Higher std across methods = higher epistemic uncertainty
        epistemic = np.mean(prob_std) + disagreement * 0.5
        epistemic = min(epistemic, 1.0)
        
        # 5. Aleatoric uncertainty (entropy of mean prediction)
        entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-10))
        max_entropy = np.log(len(mean_probs))
        aleatoric = entropy / max_entropy
        
        # 6. Total uncertainty
        total = np.sqrt(epistemic ** 2 + aleatoric ** 2) / np.sqrt(2)
        
        # 7. Predicted class and confidence
        predicted_class = np.argmax(mean_probs)
        raw_confidence = float(mean_probs[predicted_class])
        
        # 8. Calibrated confidence
        calibrated_probs = self.calibrator.calibrate_probs(mean_probs.reshape(1, -1))[0]
        calibrated_confidence = float(calibrated_probs[predicted_class])
        
        # 9. Mutual information (epistemic component)
        # MI = H(E[p]) - E[H(p)]
        expected_entropy = np.mean([
            -np.sum(p * np.log(p + 1e-10)) 
            for p in [mc_mean, ens_mean]
        ])
        mutual_info = entropy - expected_entropy
        mutual_info = max(0, mutual_info / max_entropy)  # Normalize
        
        # 10. OOD detection
        if features is not None:
            ood_score, is_ood = self.ood_detector.detect(
                features.reshape(1, -1), 
                np.log(mean_probs + 1e-10).reshape(1, -1)
            )
            ood_score = float(ood_score[0])
            is_ood = bool(is_ood[0])
        else:
            ood_score = 0.0
            is_ood = False
        
        # 11. Reliability score
        reliability = (
            0.3 * (1 - epistemic) +
            0.3 * calibrated_confidence +
            0.2 * (1 - disagreement) +
            0.2 * (1 - ood_score)
        )
        is_reliable = reliability >= self.reliability_threshold and not is_ood
        
        # 12. Prediction interval (95% CI)
        z = 1.96  # 95% CI
        ci_width = z * prob_std[predicted_class]
        lower_bound = max(0, mean_probs[predicted_class] - ci_width)
        upper_bound = min(1, mean_probs[predicted_class] + ci_width)
        
        return UncertaintyEstimate(
            mean_probs=mean_probs,
            prob_std=prob_std,
            predicted_class=int(predicted_class),
            prediction_confidence=raw_confidence,
            epistemic=float(epistemic),
            aleatoric=float(aleatoric),
            total=float(total),
            calibrated_confidence=float(calibrated_confidence),
            entropy=float(entropy / max_entropy),
            mutual_info=float(mutual_info),
            is_reliable=is_reliable,
            reliability_score=float(reliability),
            lower_bound=float(lower_bound),
            upper_bound=float(upper_bound),
            ood_score=float(ood_score),
            is_ood=is_ood
        )
    
    def should_trade(self, estimate: UncertaintyEstimate, 
                     min_confidence: float = 0.55,
                     max_uncertainty: float = 0.5) -> Tuple[bool, str]:
        """
        Determine if we should trade based on uncertainty
        
        Returns:
            should_trade: Boolean
            reason: Explanation
        """
        if estimate.is_ood:
            return False, "Out-of-distribution input - unusual market conditions"
        
        if not estimate.is_reliable:
            return False, f"Low reliability score: {estimate.reliability_score:.2%}"
        
        if estimate.calibrated_confidence < min_confidence:
            return False, f"Low confidence: {estimate.calibrated_confidence:.2%} < {min_confidence:.2%}"
        
        if estimate.total > max_uncertainty:
            return False, f"High uncertainty: {estimate.total:.2%} > {max_uncertainty:.2%}"
        
        if estimate.epistemic > 0.4:
            return False, f"High model uncertainty: {estimate.epistemic:.2%}"
        
        return True, f"Confidence: {estimate.calibrated_confidence:.2%}, Uncertainty: {estimate.total:.2%}"
    
    def get_position_multiplier(self, estimate: UncertaintyEstimate) -> float:
        """
        Get position size multiplier based on uncertainty
        
        Returns value between 0 and 1 to scale position size
        """
        if not estimate.is_reliable:
            return 0.0
        
        # Base on reliability score
        base = estimate.reliability_score
        
        # Reduce for high uncertainty
        uncertainty_factor = 1 - estimate.total
        
        # Reduce for low confidence
        confidence_factor = estimate.calibrated_confidence
        
        # Combined
        multiplier = base * 0.4 + uncertainty_factor * 0.3 + confidence_factor * 0.3
        
        return float(np.clip(multiplier, 0, 1))


class CalibrationMetrics:
    """
    Metrics to evaluate calibration quality
    """
    
    @staticmethod
    def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, 
                                   n_bins: int = 10) -> float:
        """
        Expected Calibration Error (ECE)
        Lower is better (0 = perfectly calibrated)
        """
        confidences = probs.max(axis=1)
        predictions = probs.argmax(axis=1)
        accuracies = (predictions == labels).astype(float)
        
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        
        for i in range(n_bins):
            in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                avg_confidence = confidences[in_bin].mean()
                avg_accuracy = accuracies[in_bin].mean()
                ece += np.abs(avg_accuracy - avg_confidence) * prop_in_bin
        
        return ece
    
    @staticmethod
    def maximum_calibration_error(probs: np.ndarray, labels: np.ndarray,
                                  n_bins: int = 10) -> float:
        """
        Maximum Calibration Error (MCE)
        """
        confidences = probs.max(axis=1)
        predictions = probs.argmax(axis=1)
        accuracies = (predictions == labels).astype(float)
        
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        mce = 0.0
        
        for i in range(n_bins):
            in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
            
            if in_bin.sum() > 0:
                avg_confidence = confidences[in_bin].mean()
                avg_accuracy = accuracies[in_bin].mean()
                mce = max(mce, np.abs(avg_accuracy - avg_confidence))
        
        return mce
    
    @staticmethod
    def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
        """
        Brier Score (mean squared error of probabilities)
        Lower is better
        """
        one_hot = np.zeros_like(probs)
        one_hot[np.arange(len(labels)), labels] = 1
        return np.mean((probs - one_hot) ** 2)
    
    @staticmethod
    def reliability_diagram(probs: np.ndarray, labels: np.ndarray, 
                           n_bins: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Data for reliability diagram
        
        Returns:
            bin_centers: Center of each bin
            accuracies: Accuracy in each bin
            counts: Number of samples in each bin
        """
        confidences = probs.max(axis=1)
        predictions = probs.argmax(axis=1)
        correct = (predictions == labels).astype(float)
        
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_centers = []
        accuracies = []
        counts = []
        
        for i in range(n_bins):
            in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
            count = in_bin.sum()
            
            if count > 0:
                bin_centers.append((bin_boundaries[i] + bin_boundaries[i+1]) / 2)
                accuracies.append(correct[in_bin].mean())
                counts.append(count)
        
        return np.array(bin_centers), np.array(accuracies), np.array(counts)