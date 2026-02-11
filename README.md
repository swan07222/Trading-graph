🤖 Production-Grade AI Trading System (A-Share Optimized)
"A Self-Driving Car for the Stock Market"
This project is a sophisticated, institutional-grade algorithmic trading system designed for the Chinese A-Share market. Unlike simple trading scripts that rely on a single indicator (like RSI), this system uses an Ensemble of Deep Learning Models to predict price movements, reads financial news to gauge sentiment, and employs strict risk management to protect capital.

📖 How It Works (The Logic Flow)
The system operates on a continuous "Observe-Orient-Decide-Act" loop:

👀 Observe (Data Ingestion):

It pulls real-time price data using a smart fetcher that automatically detects if you are using a VPN (switching between AkShare, Tencent, and Yahoo Finance accordingly).
It scrapes financial news sources (Sina, Eastmoney) to detect market sentiment.
🧠 Think (AI Processing):

Raw data is converted into technical indicators (MACD, Bollinger Bands, ATR).
This data is fed into an Ensemble of 5 Neural Networks (LSTM, GRU, Transformer, TCN, and a Hybrid CNN).
The models "vote" on the probability of the price going UP, DOWN, or Sideways.
🛡️ Protect (Risk Management):

Before buying, the Risk Manager checks: "Do we have too much money in this sector?", "Is the market crashing?", "Is this data stale?".
If the risk is too high, the Kill Switch activates and blocks the trade.
⚡ Act (Execution):

If the AI is confident (>60%) and Risk checks pass, the Executor sends an order.
It uses an Order Management System (OMS) to track the trade locally (Double-entry bookkeeping) to ensure no money is ever "lost" due to software crashes.
🎓 Deep Dive: The "Auto-Learner" Engine
The most unique feature of this system is the Continuous Auto-Learner (models/auto_learner.py). It allows the system to run indefinitely, constantly finding new opportunities without human intervention. Here is exactly how it acts, cycle by cycle:

The "School Class" Analogy
Imagine a teacher (The Learner) who has thousands of students (Stocks) but a small classroom (GPU Memory).

Cycle 1 (Discovery):

The system scans the market and picks a random batch of 50 new stocks.
It downloads their history and trains the AI models on this specific batch.
The Exam: It tests how well the AI predicts these 50 stocks on "Holdout Data" (data the AI has never seen).
Grading:
Stocks where the AI predicted correctly are marked as "A-Students" and saved into the Experience Replay Buffer.
Stocks where the AI failed are marked as "Failed" and discarded.
Cycle 2 (Rotation):

The system picks 50 DIFFERENT stocks that it hasn't seen yet.
Crucially, it mixes in a few "A-Students" from Cycle 1 to make sure the AI doesn't forget what it learned previously.
It retrains the models on this new mixed batch.
The Result (Accumulated Knowledge):

Over time, the Experience Replay Buffer fills up with only the most predictable stocks in the market.
The AI stops wasting time on unpredictable, chaotic stocks and focuses its computing power on the ones it "understands" best.
This creates a personalized "Stock Universe" tailored to your specific AI model's strengths.
📂 File Structure & Explanations
1. Root Directory
main.py: The commander. It checks dependencies, sets up the environment, and launches the mode you selected (UI, Training, or Headless Trading).
2. analysis/ (The Analysts)
backtest.py: A time machine. It runs your strategy over past data using a "Walk-Forward" method (training on Jan, testing on Feb; training on Jan-Feb, testing on Mar) to ensure realistic results.
technical.py: The mathematician. Calculates RSI, MACD, Support/Resistance levels, and Trend direction.
sentiment.py: The reader. Uses keyword analysis (and optional BERT models) to score news headlines as Positive or Negative.
3. config/ (The Control Panel)
settings.py: The single source of truth. Contains settings for Risk limits, AI hyperparameters, and File paths. It ensures you don't have "magic numbers" scattered in your code.
4. core/ (The Nervous System)
events.py: The messaging system. It allows different parts of the app (e.g., Data Fetcher -> Risk Manager) to talk to each other without being tightly coupled.
types.py: Defines the "language" of the system. What exactly does an "Order" look like? What is a "Position"?
network.py: The internet sensor. Detects if you are in China or using a VPN and routes data requests to the fastest server (Eastmoney vs Yahoo).
5. data/ (The Fuel)
fetcher.py: The heavy lifter. Downloads price data. It handles retries, rate limits, and caching so you don't get banned by data providers.
processor.py: The refinery. Cleans dirty data, handles missing values, and scales numbers (0 to 1) for the AI. Critical: It implements an "Embargo" to prevent data leakage during training.
database.py: The memory. A SQLite wrapper that stores price history locally to make the app faster and offline-capable.
news.py: The news aggregator. Fetches, deduplicates, and caches news articles.
6. models/ (The Brain)
networks.py: The blueprints. Defines the actual architecture of the neural networks (LSTM, TCN, Transformer).
ensemble.py: The manager. It trains all the networks separately and combines their votes. It uses "Temperature Scaling" to ensure the confidence score (e.g., "80% sure") is actually accurate.
auto_learner.py: The teacher. Manages the continuous learning loops, stock rotation, and model backups.
predictor.py: The translator. Takes the raw output from the AI (e.g., [0.1, 0.2, 0.7]) and converts it into a human-readable signal (e.g., "STRONG BUY").
7. trading/ (The Business Logic)
oms.py (Order Management System): The accountant. Maintains the local database of your money and shares. It handles the "T+1" settlement rule valid in China.
risk.py: The bodyguard. Checks every order against rules like "Max 15% of portfolio in one stock" or "Stop trading if down 3% today."
kill_switch.py: The emergency brake. Monitors system health and market crashes to halt trading instantly.
executor.py: The coordinator. Takes a signal, checks with Risk Manager, reserves funds in OMS, and sends the order to the Broker.
broker.py: The interface. Connects to the trading software (via easytrader or simulation) to place actual orders.
8. ui/ (The Face)
app.py: The main window. Assembles the dashboard, charts, and control panels.
charts.py: A high-performance charting widget (using pyqtgraph) to visualize prices and AI predictions.
news_widget.py: Displays the live news feed with color-coded sentiment analysis.
9. utils/ (The Toolbelt)
security.py: Encrypts your passwords and logs every action for auditing.
atomic_io.py: Ensures files are saved safely. It writes to a temporary file first, then swaps it, so a power outage won't corrupt your data.
metrics.py: Exposes system health data (CPU usage, trade latency) to Prometheus for professional monitoring.
🚀 Getting Started
Prerequisites
Python 3.9+
TA-Lib (Technical Analysis Library)
Installation
Bash

# 1. Install Python dependencies
pip install -r requirements.txt

# 2. (Optional) Install PyTorch with CUDA for GPU acceleration
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
Running the System
1. Launch the Desktop GUI:

Bash

python main.py
2. Start Auto-Learning Mode (Headless/Server):

Bash

python main.py --auto-learn --max-stocks 200 --continuous
3. Run a Backtest:

Bash

python main.py --backtest
4. Check System Health:

Bash

python main.py --health