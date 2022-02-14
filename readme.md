# Ethereum-Forecaster
Ethereum-Forecaster is a stacked LSTM model with 3, 50 unit layers that forecasts ETH-CAD for the month of February. As of Feb 14, 2022, it has accurately predicted the dip in price, and predicts that ETH will hit nearly $3,000 CAD by the end of the month. Data was scraped from Yahoo Finance, preprocessed using pandas and scikit-learn, while the model was generated using TF Keras, and plotting was done with MatPlotLib. The model was training on an AWS EC2 instance.

The blue graph entails the training data while the orange is the output of the model on training data:
![plot](./images/train.png)

The following graph is the predicted price of ETH in CAD until the end of February. The prediction starts from '450' on the x-axis.
![plot](./images/predicted.png)
