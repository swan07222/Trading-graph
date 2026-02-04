# 🤖 AI Stock Trading Advisor (A股智能交易顾问)

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An advanced AI-powered stock trading system for Chinese A-share market with real broker integration.

![Main Interface](docs/images/main_ui.png)

## ✨ Features

### 🧠 AI Prediction Engine
- **6 Neural Network Models**: LSTM, Transformer, GRU, TCN, Mamba, Hybrid
- **Ensemble Learning**: Combines all models for robust predictions
- **Uncertainty Quantification**: Knows when predictions are unreliable
- **80+ Technical Features**: Comprehensive market analysis

### 📊 Technical Analysis
- RSI, MACD, Bollinger Bands, Ichimoku Cloud
- Support/Resistance detection
- Candlestick pattern recognition
- Volume analysis

### 📰 Sentiment Analysis
- Chinese financial news scraping (新浪财经, 东方财富, 财联社)
- Keyword-based sentiment scoring
- Optional BERT-based analysis

### 💹 Real Trading
- **Simulation Mode**: Paper trading with realistic execution
- **Live Trading**: 同花顺 (TongHuaShun) integration
- **Risk Management**: Position sizing, stop-loss, daily limits

### 🖥️ Professional GUI
- Real-time charts with predictions
- Portfolio tracking
- Order management
- Trading signals dashboard

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ai-stock-advisor.git
cd ai-stock-advisor

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Verify installation
python main.py --check