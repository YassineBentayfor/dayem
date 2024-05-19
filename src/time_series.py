import pandas as pd
from prophet import Prophet

class ProphetModel:
    """
    A class to perform time series forecasting using the Prophet library.

    Attributes:
    - data_paths (list): List of file paths containing the time series data.
    - df (DataFrame): Concatenated DataFrame of all input data files.
    - df_prophet1 (DataFrame): Processed DataFrame for Prophet forecasting (temperature).
    - df_prophet2 (DataFrame): Processed DataFrame for Prophet forecasting (humidity).
    - df_prophet3 (DataFrame): Processed DataFrame for Prophet forecasting (precipitation).
    """

    def __init__(self, data_paths):
        """
        Initialize the ProphetModel with the given data paths.

        Parameters:
        - data_paths (list): List of file paths containing the time series data.
        """
        data1 = pd.read_csv(data_paths[0])
        data2 = pd.read_csv(data_paths[1])
        data3 = pd.read_csv(data_paths[2])
        self.df = pd.concat([data1, data2, data3], axis=0)
        self._prepare_data()

    def _prepare_data(self):
        self.df_prophet1 = self.df.loc[:, ['datetime', 'temp']].rename(columns={'datetime': 'ds', 'temp': 'y'})
        self.df_prophet2 = self.df.loc[:, ['datetime', 'humidity']].rename(columns={'datetime': 'ds', 'humidity': 'y'})
        self.df_prophet3 = self.df.loc[:, ['datetime', 'precip']].rename(columns={'datetime': 'ds', 'precip': 'y'})
        for df in [self.df_prophet1, self.df_prophet2, self.df_prophet3]:
            df.dropna(inplace=True)
            df['ds'] = pd.to_datetime(df['ds'])

    def _fit_model(self, train_data):
        model = Prophet()
        model.fit(train_data)
        return model

    def _forecast(self, model, periods):
        future = model.make_future_dataframe(periods=periods)
        return model.predict(future)

    def run(self):
        results = []
        for df_prophet in [self.df_prophet1, self.df_prophet2, self.df_prophet3]:
            train_size = int(0.85 * len(df_prophet))
            train_data = df_prophet[:train_size]
            test_data = df_prophet[train_size:]

            model = self._fit_model(train_data)
            forecast = self._forecast(model, len(test_data))

            error = forecast.iloc[train_size:]['yhat'] - test_data['y']
            mse = (error ** 2).mean()
            results.append((model, forecast, mse))
            print("MSE = ", mse)

        return results


import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchtsmixer import TSMixer

class TSMixerModel:
    """
    A class to perform time series forecasting using the TSMixer library.

    Attributes:
    - data_paths (list): List of file paths containing the time series data.
    - df (DataFrame): Concatenated DataFrame of all input data files.
    - sequence_length (int): Length of input sequences for the model.
    - prediction_length (int): Length of output sequences for the model.
    - input_channels (int): Number of input channels for the model.
    - output_channels (int): Number of output channels for the model.
    """

    def __init__(self, data_paths):
        """
        Initialize the TSMixerModel with the given data paths.

        Parameters:
        - data_paths (list): List of file paths containing the time series data.
        """
        data1 = pd.read_csv(data_paths[0])
        data2 = pd.read_csv(data_paths[1])
        data3 = pd.read_csv(data_paths[2])
        self.df = pd.concat([data1, data2, data3], axis=0)
        self.df.reset_index(drop=True, inplace=True)
        self.df['datetime'] = pd.to_datetime(self.df['datetime'])
        self.df.set_index('datetime', inplace=True)
        self.data = self.df[['humidity', 'temp', 'precip']]
        self.sequence_length = 10
        self.prediction_length = 5
        self.input_channels = 3
        self.output_channels = 3

    def _prepare_data(self):
        data = torch.tensor(self.data.to_numpy())[:720].reshape((-1, self.sequence_length + self.prediction_length, self.input_channels))
        X, y = data[:, :self.sequence_length, :], data[:, self.sequence_length:, :]
        gap = 35
        X_train, X_test = X[:-gap], X[-gap:]
        y_train, y_test = y[:-gap], y[-gap:]
        return X_train, X_test, y_train, y_test

    def _train_model(self, X_train, y_train):
        model = TSMixer(self.sequence_length, self.prediction_length, self.input_channels, self.output_channels)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.1)

        num_epochs = 5000
        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train.float())
            loss = criterion(outputs, y_train.float())
            loss.backward()
            optimizer.step()
            if epoch % 50 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
        return model

    def run(self):
        X_train, X_test, y_train, y_test = self._prepare_data()
        model = self._train_model(X_train, y_train)

        outputs_test = model(X_test.float())
        for i in range(3):
            plt.figure()
            stack_y, stack_pred = [], []
            counter = 0
            while True:
                stack_y.append(y_test[counter, :, i])
                stack_pred.append(outputs_test[counter, :, i])
                counter += 5
                if counter >= y_test.shape[0]:
                    break
            groundtruth = torch.stack(stack_y, dim=0).flatten()
            prediction = torch.stack(stack_pred, dim=0).flatten()
            plt.plot(groundtruth, label='groundtruth')
            plt.plot(prediction.detach().cpu().numpy(), label='prediction')
            plt.legend()
        plt.show()



import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

class ARIMA_Model:
    """
    A class to perform time series forecasting using the ARIMA model.

    Attributes:
    - data_paths (list): List of file paths containing the time series data.
    - column_name (str): Name of the column to be forecasted.
    - data (DataFrame): Preprocessed DataFrame with resampled time series data.
    """

    def __init__(self, data_paths, column_name):
        """
        Initialize the ARIMA_Model with the given data paths and column name.

        Parameters:
        - data_paths (list): List of file paths containing the time series data.
        - column_name (str): Name of the column to be forecasted.
        """
        data1 = pd.read_csv(data_paths[0], usecols=['datetime', column_name])
        data2 = pd.read_csv(data_paths[1], usecols=['datetime', column_name])
        data3 = pd.read_csv(data_paths[2], usecols=['datetime', column_name])
        self.data = pd.concat([data1, data2, data3], axis=0)
        self.data['datetime'] = pd.to_datetime(self.data['datetime'])
        self.data.set_index('datetime', inplace=True)
        self.data = self.data.resample('D').mean().fillna(method='ffill')
        self.column_name = column_name

    def _split_data(self):
        train_size = int(len(self.data) * 0.8)
        train, test = self.data[:train_size], self.data[train_size:]
        return train, test

    def _fit_model(self, train):
        model = ARIMA(train[self.column_name], order=(150, 1, 0))
        return model.fit()

    def run(self):
        train, test = self._split_data()
        model_fit = self._fit_model(train)
        forecast = model_fit.forecast(steps=len(test))
        forecast_index = pd.date_range(start=test.index[0], periods=len(test), freq='D')
        forecast_df = pd.DataFrame(forecast, index=forecast_index, columns=[self.column_name])

        plt.figure(figsize=(10, 6))
        plt.plot(train.index, train[self.column_name], label='Training Data')
        plt.plot(test.index, test[self.column_name], label='Test Data')
        plt.plot(test.index, forecast, label='ARIMA Forecast')
        plt.title(f'ARIMA {self.column_name.capitalize()} Forecasting')
        plt.xlabel('Date')
        plt.ylabel(self.column_name.capitalize())
        plt.legend()
        plt.show()

        return forecast_df
