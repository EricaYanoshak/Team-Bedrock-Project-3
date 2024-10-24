# Stock Price Forecasting Using Deep Learning and Equity Analysis

**Team Bedrock**: Brandon Rusinque, Rasesh Desai, and Erica Yanoshak

## Overview
This project combines traditional equity trading analysis with advanced AI-driven deep learning models to predict stock prices for three major publicly traded companies: Tesla, Apple, and Nvidia. The project aims to assess how accurately stock prices can be predicted using a deep learning Long Short-Term Memory (LSTM) model and, optionally, explore the influence of sentiment analysis from news and earnings calls on prediction accuracy.

### Key Questions
* How accurately can the deep learning model forecast short-term stock prices?
* Can sentiment analysis from news and earnings calls enhance prediction accuracy?
* How do market factors such as trading volume and volatility influence stock price predictions?

### Data Sources
* **Historical Stock Data**: Retrieved from Yahoo Finance or Kaggle, covering at least one year of data for Tesla, Apple, and Nvidia.
* **Sentiment Data**: Collected via APIs like Alpha Vantage or FinViz for news and earnings call sentiment.

### Tools and Technologies
* **Programming Languages**: Python
* **Data Libraries**:
  * Pandas (Data manipulation)
  * Matplotlib & Seaborn (Visualization)
  * Deep Learning Framework:
  * TensorFlow & Keras (LSTM model development)
  * KerasTuner (Hyperparameter optimization)
* **NLP Libraries**:
  * Hugging Face Transformers (For NLP sentiment analysis)
  * spaCy or VADER (For sentiment scoring)
  * Whisper (For transcribing earnings calls, if used)
* **Version Control**: Git/GitHub for code management
* **Collaboration**:
  * GitHub Projects (Task tracking and project management)

## Project Workflow
1. **Data Collection & Preprocessing**:
  * Collect historical stock price data and sentiment data
  * Clean and preprocess data, handling missing values and scaling the data for model training
  * Tokenize and preprocess sentiment data (if applicable)

2. **Exploratory Data Analysis (EDA)**:
  * Conduct equity trading analysis using visualizations such as line graphs, heatmaps, and Bollinger Bands to explore trends and volatility
  * Analyze the correlation between stock prices, trading volume, and market sentiment (if applicable)

3. **Model Development**:
  * Build and train the LSTM model using stock price and market data
  * Integrate sentiment data (optional) for enhanced forecasting

4. **Model Evaluation**:
  * Evaluate the model using MSE, RMSE, and MAE
  * Visualize predicted vs actual stock prices

5. **Sentiment Analysis Integration**:
  * Incorporate sentiment data and assess its impact on prediction accuracy
  * Compare model performance with and without sentiment analysis

6. **Model Deployment**:
  * Deploy the model with an interface (e.g., Gradio) to provide stock predictions based on user input.

## Project Structure
* **/data**: Contains raw and processed datasets
* **/notebooks**: Jupyter notebooks for EDA, model training, and evaluation
* **/models**: Saved models and related artifacts
* **/src**: Python scripts for data preprocessing, model training, and evaluation
* **README.md**: This file, containing the project overview and instructions

## Installation & Setup
1. Clone the repository: `git clone https://github.com/TeamBedrock/StockPriceForecasting.git`
2. Install dependencies:
    `pip install -r requirements.txt`
3. Download datasets and place them in the `/data` folder
4. Run Jupyter notebooks in `/notebooks` to explore data and train the models

### Usage
* Run the `data_preprocessing.py` script in the `/src` folder to preprocess data
* Train the LSTM model by running `model_training.py`
* Evaluate the model using the metrics provided in the notebooks

## Results and Conclusion 
### Results
* Visualizations of stock trends and volatility
* Predicted vs actual stock prices
* Impact of sentiment analysis on model predictions

### Conclusion
This project evaluates the impact of sentiment on stock performance, incorporating sentiment analysis from news, social media, and financial reports. By factoring in the emotional and psychological responses of the market, this framework provides a more holistic understanding of stock movements. Overall, this integration of historical analysis, predictive modeling, and sentiment analysis creates a robust tool for modern stock analysis and decision-making.


## Insights and Next Steps
### Noteworthy Pain Points and Learning Experiences
Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. 

### Future Work & Next Steps:
* Enhance the model by including more companies and extending the prediction period
* Improve sentiment analysis by integrating more sophisticated NLP techniques
* Deploy the model as a web app for user-friendly interaction


## License
This project is licensed under the [MIT License](./LICENSE.txt)


## Libraries Used:
Special thanks to the open-source community for providing these powerful tools.
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn
* Imbalanced-Learn
* XGBoost
* Plotly
* Dash


## Resources Consulted 
* AI Bootcamp, Models 18 -22. (2023). edX Boot Camps LLC

## Acknowledgements
* This project took advantage of [Xpert Learning Assistant](https://bootcampspot.instructure.com/courses/6141/external_tools/313) to help with coding errors.
* This project utilized assistance from [ChatGPT](https://openai.com/chatgpt), an AI language model developed by OpenAI, for generating code snippets, explanations, and guidance.










