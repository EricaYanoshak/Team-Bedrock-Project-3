# Stock Price Forecasting Using Deep Learning and Equity Analysis

**Team Bedrock**: Brandon Rusinque, Rasesh Desai, and Erica Yanoshak

## Overview
This project explores the use of deep learning models to forecast stock prices and evaluate market trends. Using an LSTM (Long Short-Term Memory) neural network architecture, we analyze and predict short-term stock prices for major companies, focusing on Microsoft, Tesla, and Google. The project also integrates interactive visualizations to enhance the analysis and presentation of results.

### Key Questions
1. How accurately can the deep learning model forecast short-term stock prices?
2. Can sentiment analysis from news and earnings calls enhance prediction accuracy?
3. How do market factors such as trading volume and volatility influence stock price predictions?

### Objectives
* Develop an LSTM-based deep learning model to predict short-term stock prices.
* Integrate historical stock data and advanced data preprocessing techniques to ensure the reliability of results.
* Evaluate the model's performance using metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), and Test Loss.
* Create interactive visualizations to support an intuitive exploration of the forecasting results.

### Tools and Technologies
* Data Collection:
  * **Alpha Vantage API** for historical stock data retrieval
* Development Environment:
  * **Jupyter Notebook** for coding and interactive data exploration
* Version Control:
  * **GitHub** for project collaboration and code versioning
* Visualization Frameworks:
  **Plotly Dash** for building interactive web-based visualizations

## Project Workflow
### Data Collection & Preprocessing
* **Data Sources**
  * Stock data was collected using the Alpha Vantage API, which provided daily adjusted stock price data.
* **Data Processing**:
  * Historical data for Microsoft (MSFT), Tesla (TSLA), and Google (GOOGL) was processed and cleaned.
  * Missing values were handled, and the data was normalized using the MinMaxScaler to scale the features to a range between 0 and 1.
  * Moving averages (50-day and 200-day) were calculated and included as indicators for trend analysis.

### Model Development
* **Architecture**
  * The model uses an LSTM neural network with dropout layers to reduce overfitting and improve generalization.
* **Training and Evaluation**
  * The model was trained with a sequence length of 60 days to predict future stock prices.
  * The training process included a validation split to monitor and adjust the model's performance.
* **Performance Metrics**
  * Test Loss: 0.0664
  * Mean Absolute Error (MAE): 0.1073
  * Mean Squared Error (MSE): 0.0125

### Interactive Visualizations
The project incorporates interactive visualizations to enhance data exploration and analysis:
  * Line plots with confidence intervals to compare actual vs. predicted prices.
  * Residual plots to show the distribution of prediction errors.
  * Interactive dashboards built using Plotly Dash for user-friendly exploration of stock trends and volume data.

## Results and Conclusions
The results demonstrate that the LSTM model captures short-term stock price trends with reasonable accuracy. While the model's predictions align closely with actual stock prices, deviations indicate the potential for further optimization. The use of deep learning for stock forecasting proves valuable, though integration with external data, such as market sentiment, may further enhance prediction reliability.

#### 1. Line Plot with Confidence Interval
* Shows the actual vs. predicted prices as line plots with a confidence interval shaded area around the predicted values.
    
![1LinePlotwConfidenceInterval.png](./Images/1LinePlotwConfidenceInterval.png)


#### 2. Cumulative Returns Plot for Actual vs Predicted
* Plotting cumulative returns based on actual and predicted prices helps show the model matches the trends over time.
  
![2ReturnsActualvsPredicted.png](./Images/2ReturnsActualvsPredicted.png)  


#### 3. Density Plot of Daily Returns
* Shows the probability distribution of daily returns, useful for understanding stock volatility.

![Density Plot of Daily Returns.png](./Images/3DailyReturns.png)


#### 4. Rolling Mean of Absolute Prediction Errors
* Provides insights into trends in the prediction error over time, smoothing out short-term fluctuations.
  
![RollingMeanPredictionErrors.png](./Images/4RollingMeanPredictionErrors.png)  


#### 5. Correlation Heatmap of Daily Returns
* Shows the correlation between daily returns for each stock
  
![Heatmap.png](./Images/5Heatmap.png) 


#### 6. Interactive Actual vs Predicted Prices Line Plot
* Interactive line chart to show the actual vs predicted prices.

![InteractiveLinePlot.jpg](./Images/6InteractiveLinePlot.jpg)  


#### 7. Interactive Stock Analyzer
* To create an Interactive stock analyzer, we used Plotly Dash to build a web-based interactive applications with sliders, and dropdowns, which allows users to select different stocks, view the stock’s daily prices, and add overlays like moving averages for Microsoft, Tesla, and Google.

![InteractiveStockAnalyzer.jpg](./Images/7InteractiveStockAnalyzer.jpg)

### Answers to Our Key Questions
**1. How accurately can the deep learning model forecast short-term stock prices?**
  * With a test loss of 0.0664, a mean absolute error (MAE) of 0.1073, and a mean squared error (MSE) of 0.0125, the deep learning model shows a moderate level of accuracy in forecasting short-term stock prices. The relatively low MSE indicates that the model's predictions are, on average, close to the actual values, while the MAE provides a straightforward interpretation of the average prediction error. 

**2. Can sentiment analysis from news and earnings calls enhance prediction accuracy?**
  * Incorporating sentiment analysis from news articles and earnings calls can potentially enhance prediction accuracy by providing insights into market sentiment and investor behavior, which are influential factors in stock price movements.
    
**3. How do market factors such as trading volume and volatility influence stock price predictions?** 
  * Market factors such as trading volume and volatility significantly impact stock price predictions; high trading volumes often indicate strong investor interest, while increased volatility reflects greater uncertainty, both of which can affect the model's predictive performance.

The results highlight that while the model is useful for providing forecasts that align with actual trends, it may not capture all market nuances, suggesting room for further refinement.


### Conclusion
The conclusion drawn from these results suggests that deep learning models, specifically those using LSTM architecture, can be valuable tools for short-term stock price forecasting. However, integrating additional data sources, such as sentiment analysis or broader market indicators, might enhance predictive performance and mitigate errors, thus offering more reliable guidance for practical investment decisions.


## Future Work & Next Steps
* **Sentiment Analysis**: Integrate sentiment analysis from financial news and earnings call transcripts to incorporate market sentiment into predictions.
* **Ensemble Methods**: Explore hybrid models that combine LSTM with traditional statistical methods to capture diverse data patterns.
* **Model Deployment**: Create a user-friendly web application for real-time forecasting and easy user interaction.
* **Hyperparameter Tuning**: Use more advanced strategies for hyperparameter optimization to improve model accuracy.

## Libraries Used:
Special thanks to the open-source community for providing these powerful tools.
* **Python Standard Libraries**: `os, time, datetime`
* **Data Collection**: `requests` for API calls
* **Data Manipulation**: `pandas` for data handling and DataFrame operations
* **Data Visualization**:
  * `matplotlib.pyplot` for basic plotting
  * `seaborn` for enhanced data visualization
  * `plotly` and `dash` for interactive visualizations and building dashboards
* **Machine Learning and Deep Learning**:
  * `numpy` for numerical operations
  * `tensorflow` and `keras` for building and training the LSTM model
* **Preprocessing and Scaling**:
  * `sklearn.preprocessing` (specifically `MinMaxScaler`) for feature scaling
* **Evaluation Metrics**
  * `sklearn.metrics` for calculating Mean Absolute Error (MAE) and Mean Squared Error (MSE)

## Installation & Setup
1. Clone the repository: `git clone https://github.com/Username/Team-Bedrock-Project-3.git`
2. Install the necessary libraries: `pip install -r requirements.txt`
3. Run the Jupyter Notebook or Python scripts to train the model and visualize the results.
  
## License
This project is licensed under the [MIT License](./LICENSE.txt)


## Resources Consulted 
* AI Bootcamp, Models 18 -22. (2023). edX Boot Camps LLC

## Acknowledgements
* We would like to thank the **Alpha Vantage API** for providing the data used in this project.
* This project took advantage of [Xpert Learning Assistant](https://bootcampspot.instructure.com/courses/6141/external_tools/313) to help with coding errors.
* This project utilized assistance from [ChatGPT](https://openai.com/chatgpt), an AI language model developed by OpenAI, for generating code snippets, explanations, and guidance.










