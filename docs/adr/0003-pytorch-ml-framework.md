# ADR 0003: PyTorch for ML Models

## Status
Accepted

## Date
2024-01-15  
Last Updated: 2026-02-25

## Context

The trading system requires machine learning capabilities for:
- Price prediction (direction and magnitude) - **Guessing Model (GM)**
- News sentiment analysis - **LLM Sentiment Model**
- Signal generation (buy/sell/hold)
- Pattern recognition in time series
- Ensemble modeling for robustness
- Hybrid approach combining GM + LLM

We need an ML framework that:
- Supports modern deep learning architectures (Transformer, TFT, N-BEATS)
- Has good Python integration
- Provides GPU acceleration option
- Has strong community support
- Enables model export and deployment
- Supports dynamic computation graphs for research

## Decision

Use PyTorch as the primary ML framework with the following characteristics:

### Model Architecture Overview

**Two Model Families:**

| Model | File | Architecture | Storage | Purpose |
|-------|------|--------------|---------|---------|
| **GM (Guessing Model)** | `models/ensemble.py` | Informer + TFT + N-BEATS + TSMixer | `models_saved/GM/` | Price prediction |
| **LLM Sentiment** | `data/llm_sentiment.py` | Transformer + MLP | `models_saved/LLM/` | Sentiment analysis |

**Guessing Model (GM) Architectures:**

| Model | File | Architecture | Purpose |
|-------|------|--------------|---------|
| **Informer** | `models/networks.py` | Probabilistic attention | Long sequence forecasting O(L log L) |
| **TFT** | `models/networks.py` | Temporal Fusion Transformer | Interpretable multi-horizon predictions |
| **N-BEATS** | `models/networks.py` | Neural basis expansion | Trend and seasonality decomposition |
| **TSMixer** | `models/networks.py` | All-MLP architecture | Efficient time series mixing |

**LLM Sentiment Architecture:**

| Component | File | Purpose |
|-----------|------|---------|
| **Transformer Encoder** | `data/llm_sentiment.py` | Bilingual text encoding (zh/en) |
| **MLP Classifier** | `data/llm_sentiment.py` | Sentiment classification fallback |
| **Embedding Model** | `data/news_embeddings.py` | Sentence embeddings for similarity |

### Hybrid Prediction Flow

```
┌─────────────────┐    ┌─────────────────┐
│  GM (Price)     │    │  LLM (Sentiment)│
│  - Informer     │    │  - Transformer  │
│  - TFT          │    │  - MLP Fallback │
│  - N-BEATS      │    │                 │
│  - TSMixer      │    │                 │
└────────┬────────┘    └────────┬────────┘
         │                      │
         └──────────┬───────────┘
                    ▼
         ┌───────────────────┐
         │  Fusion Layer     │
         │  (Weighted Vote)  │
         └─────────┬─────────┘
                   ▼
         ┌───────────────────┐
         │  Final Prediction │
         │  (Signal + Conf)  │
         └───────────────────┘
```

### Training Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                    GM Training Pipeline                     │
│                                                             │
│  Historical Data → Features → Ensemble Models → GM/        │
│                                              ↓              │
│                                      Prediction Output      │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                   LLM Training Pipeline                     │
│                                                             │
│  News Articles → Tokenizer → Transformer → LLM/            │
│                  ↓                                         │
│            Sentiment Score                                 │
└─────────────────────────────────────────────────────────────┘
```

### Configuration

```python
# Training hyperparameters
sequence_length = 60
hidden_size = 256
num_heads = 8
dropout = 0.1
learning_rate = 1e-3
batch_size = 32
epochs = 100

# Model storage
model_dir = "models_saved"
gm_model_dir = "models_saved/GM"
llm_model_dir = "models_saved/LLM"
```

## Consequences

### Positive

- Flexible model development with dynamic computation graphs
- GPU acceleration support (CUDA, MPS)
- Strong ecosystem (torchvision, torchaudio, torchtext)
- Easy debugging with eager execution
- Good Python integration
- Active research community with latest architectures
- TorchScript for model serialization

### Negative

- Larger dependency footprint (~2GB with CUDA)
- Model versioning challenges
- Requires careful serialization for deployment
- Pickle-based model storage (security considerations)

### Mitigation

- Use TorchScript for model serialization when possible
- Implement model versioning in artifact storage
- Provide CPU-only fallback mode
- Use safe pickle loading with validation
- Store model metadata separately (config, training info)

## Implementation

### Model Definition

```python
# models/networks.py
class Informer(nn.Module):
    """Informer for long sequence forecasting."""
    
    def __init__(self, d_model, n_heads, d_ff, n_layers, seq_len, pred_len):
        super().__init__()
        self.encoder = InformerEncoder(d_model, n_heads, d_ff, n_layers)
        self.projection = nn.Linear(d_model, pred_len)
    
    def forward(self, x):
        enc_out = self.encoder(x)
        return self.projection(enc_out[:, -1])
```

### Training Loop

```python
# models/trainer.py
def train(self, epochs=100):
    for epoch in range(epochs):
        for batch in dataloader:
            optimizer.zero_grad()
            output = model(batch.x)
            loss = criterion(output, batch.y)
            loss.backward()
            optimizer.step()
```

### Model Loading

```python
# models/predictor.py
def load_ensemble(self, path):
    """Load GM ensemble model."""
    state = torch.load(path, map_location=self.device)
    
    # Validate model metadata
    self._validate_metadata(state)
    
    # Load GM models
    for name, model_state in state['models'].items():
        model = self._create_model(name)
        model.load_state_dict(model_state)
        self.models[name] = model

def load_llm_sentiment(self, path):
    """Load LLM sentiment model."""
    state = torch.load(path, map_location=self.device)
    
    # Load transformer or MLP fallback
    if 'transformer_weights' in state:
        self.llm.load_transformer_weights(state['transformer_weights'])
    if 'mlp_weights' in state:
        self.mlp.load_state_dict(state['mlp_weights'])
```

## Alternatives Considered

| Alternative | Pros | Cons | Decision |
|-------------|------|------|----------|
| **TensorFlow** | Production deployment, TFLite | Static graphs (TF1), verbose API | Rejected |
| **JAX** | Fast, functional, XLA compilation | Smaller community, less mature | Rejected |
| **MXNet** | Efficient, multi-language | Smaller ecosystem | Rejected |
| **scikit-learn** | Simple, good for traditional ML | No deep learning support | Rejected |

## References

- PyTorch Documentation: https://pytorch.org/docs/
- Informer Paper: https://arxiv.org/abs/2012.07436
- TFT Paper: https://arxiv.org/abs/1912.09363
- N-BEATS Paper: https://arxiv.org/abs/1905.10437
- TSMixer Paper: https://arxiv.org/abs/2303.06053
