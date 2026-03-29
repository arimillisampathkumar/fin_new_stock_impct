## 1. System Requirements
To replicate the environment described in the research, ensure you have Python 3.10+ installed. The following libraries are mandatory:

torch, transformers: For the FinBERT contextual sentiment extraction [2].

yfinance: For HKEX OHLCV data retrieval.

xgboost: For the gradient-boosted decision tree baseline [7].

textblob: For the subjectivity scoring component of the polyvalent vector [4, 1].

pandas, numpy, scikit-learn: For data manipulation and scaling.

## 2. Installation
Run the following command in your terminal or a Jupyter cell:

Bash
pip install torch transformers yfinance xgboost textblob pandas scikit-learn matplotlib


## 3. Execution Steps
Data Ingestion: The script uses yfinance to automatically pull data for the stratified equity cohort (Tencent, HSBC, AIA, Meituan) [1].

Sentiment Processing: The PolyvalentExtractor class initializes the ProsusAI/finbert model. On the first run, this will download approximately 400MB of model weights.

Model Training: The script defines parallel paths for LSTM and Transformer architectures. Following the research findings, the Transformer is optimized for long-range dependencies in news events ``.

Hardware Acceleration: The system automatically detects if a CUDA-enabled GPU is available to handle the Transformer overhead, which consumed 180 GPU-hours during the original study [1].

## 4. Analytical Context
The models are designed to be evaluated on the 2025 out-of-sample period, a timeframe characterized by record-breaking HKEX turnover and high volatility driven by AI-led sector surges (HKEX Group, 2026) [6]. Use the WINDOW_SIZE = 10 parameter to align with the optimal lag identified in Chapter 3.3
