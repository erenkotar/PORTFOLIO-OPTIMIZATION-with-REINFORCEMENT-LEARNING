import streamlit as st
import numpy as np
import pandas as pd
pd.options.display.float_format = '{:,.4f}'.format
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from scipy.stats import norm
from scipy.optimize import minimize
import scipy.stats
import datetime

import datetime
import warnings
warnings.filterwarnings("ignore")

yf.pdr_override()
# plt.style.use("fivethirtyeight")

def write_align(text, pos="left", style="strong"):
    st.markdown(f"<{style} style='text-align:{pos};'>{text}</{style}>",unsafe_allow_html=True)

# class Custom_yf(yf.Ticker):
#     def __init__(self, ticker):
#         super().__init__(ticker, None)
#         self.candle_data = self.history(interval="1d", start="2020-01-03")[["Open","High","Low","Close"]]

#     def candle_graph(self):
#         df = self.candle_data

#         fig = go.Figure(
#             data=go.Candlestick(x = df.index,
#             open=df["Open"],
#             close=df["Close"],
#             high=df["High"],
#             low=df["Low"]))
#         fig.update_layout(xaxis_title="Date", yaxis_title="Price (USD$)", 
#                           title=f"{self.ticker} - Candlestick Char")
#         fig.show()
        
#     def price_to_return(self):
#         returns = self.price.pct_change()
#         return returns

class Finance:
    TEMPLATE = "plotly"

    def __init__(self, tickers, start_date, end_date):
        self.tickers = [i.upper() for i in tickers]
        self.all_data = yf.Tickers(tickers).history(interval="1d", start=start_date)[["Open","High","Low","Close","Volume"]].loc[start_date:end_date]
        self.candle_datas = self.all_data[["Open","High","Low","Close"]]
        self.prices = self.candle_datas["Close"]
        self.returns = self.price_to_daily_return()
        self.cum_returns = self.returns.cumsum()

    def price_to_daily_return(self):
        returns = self.prices.pct_change()
        returns = returns.iloc[1:]
        return returns
    
    def get_one_ticker_candle_data(self, ticker_name):
        cols = [col for col in self.candle_datas.columns if col[1]== ticker_name]
        one_ticker_data = self.candle_datas[cols]
        one_ticker_data.columns = one_ticker_data.columns.droplevel(1)
        return one_ticker_data

    def get_one_ticker_price_data(self, ticker_name):
        price = self.get_one_ticker_candle_data(ticker_name)["Close"]
        return price

    # VISUALIZATIONS

    def plot_price(self):
        fig = go.Figure()
        fig.update_layout(xaxis_title="Date", yaxis_title="Price (USD$)", 
                    title=f"Price Chart", template = self.TEMPLATE)
        
        for ticker in self.tickers:
            c_data = self.all_data["Close"][[ticker]]

            fig.add_trace(go.Scatter(
                name=ticker,
                x = c_data.index,
                y = c_data[ticker],
                mode="lines"
                ))  
            
        # fig.show()
        return fig   
    
    def plot_cumreturn(self):
        fig = go.Figure()
        fig.update_layout(xaxis_title="Date", yaxis_title="Price (USD$)", 
                    title=f"Cumulative Returns Chart", template = self.TEMPLATE)
    
        for ticker in self.tickers:
            c_data = self.cum_returns[[ticker]]

            fig.add_trace(go.Scatter(
                name=ticker,
                x = c_data.index,
                y = c_data[ticker],
                mode="lines"
                ))  
            
        # fig.show()
        return fig  

    def plot_return(self):
        fig = go.Figure()
        fig.update_layout(xaxis_title="Date", yaxis_title="Price (USD$)", 
                    title=f"Return Chart", template = self.TEMPLATE)
        
        for ticker in self.tickers:
            c_data = self.returns[[ticker]]

            fig.add_trace(go.Scatter(
                name=ticker,
                x = c_data.index,
                y = c_data[ticker],
                mode="lines"
                ))  
            
        # fig.show()
        return fig   
            
    def candlestick_graph(self):
        fig = go.Figure()
        fig.update_layout(xaxis_title="Date", yaxis_title="Price (USD$)", 
                    title=f"Daily Candlestick Chart")
        
        for ticker in self.tickers:
            c_data = self.get_one_ticker_candle_data(ticker)

            fig.add_trace(go.Candlestick(
                name=ticker,
                x = c_data.index,
                open=c_data["Open"],
                close=c_data["Close"],
                high=c_data["High"],
                low=c_data["Low"],
                ))      
            
        # fig.show()
        return fig
    
    def histogram_graph(self):
        fig = go.Figure()
        fig.update_layout(xaxis_title="Return", yaxis_title="Frequency", 
                    title=f"Return Histogram")
        
        for ticker in self.tickers:
            c_data = self.get_one_ticker_price_data(ticker)

            fig.add_trace(go.Histogram(
                name=ticker,
                x = c_data.pct_change(),
                ))      
            
        # fig.show()
        return fig
    
    # ANALYSIS

    def conduct_all_analysis(self, period=12, level=0.05, rfree_rate=0.001, modified=False):

        annr_s = self.annualize_rets(periods_per_year=period)
        annv_s = self.annualize_vol(periods_per_year=period)
        skewness_s = self.skewness()
        kurtosis_s = self.kurtosis()
        is_normal_s = self.is_normal(level=level)
        sharpe_ratio_s = self.sharpe_ratio(riskfree_rate=rfree_rate, periods_per_year=period)
        max_drawdown_s = -self.drawdown_f().min(axis=0)
        semideviation_s = self.semideviation()
        var_historic_s = self.VaR_historic(level=level)
        var_gaussian_s = self.VaR_gaussian(level=level, modified=modified)
        
        all_in = [annr_s,annv_s,skewness_s,kurtosis_s,is_normal_s,sharpe_ratio_s,max_drawdown_s,semideviation_s,var_historic_s,var_gaussian_s]
        all_in_df = pd.concat(all_in, axis=1)
        cols = [f"Anualized Returns (period:{period})", f"Annualized Volatilities (period:{period})", "Skewness (norm:0)", "Kurtosis (norm:3)", 
                f"Is Normal (p_value:{level})", f"Sharpe Ratio (period:{period}, risk-free rate:{rfree_rate})","Max Drawndwon","Semi Deviation", 
                f"Value at Risk by Historic (percentile:{level})", f"Value at Risk by Gaussian (percentile:{level}, modified:{modified})"]
        all_in_df.columns=cols
        all_in_df = all_in_df.T
        return all_in_df

    def check_validity(price):
        pass

    def annualize_rets(self, r=None ,periods_per_year=12):
        """
        Annualizes a set of returns
        We should infer the periods per year
        but that is currently left as an exercise
        to the reader :-)
        """
        if isinstance(r, type(None)):
            r = self.returns
        else:
            r = r

        compounded_growth = (1+r).prod()
        n_periods = r.shape[0]
        return compounded_growth**(periods_per_year/n_periods)-1

    def annualize_vol(self, r=None, periods_per_year=12):
        """
        Annualizes the vol of a set of returns
        We should infer the periods per year
        but that is currently left as an exercise
        to the reader :-)
        """
        if isinstance(r, type(None)):
            r = self.returns
        else:
            r = r
        return r.std()*(periods_per_year**0.5)

    def drawdown_f(self):
        """Takes a time series of asset returns.
        returns a DataFrame with columns for
        the wealth index, 
        the previous peaks, and 
        the percentage drawdown
        """

        return_series = self.returns
        wealth_index = (1+return_series).cumprod()
        previous_peaks = wealth_index.cummax()
        drawdowns = (wealth_index - previous_peaks)/previous_peaks

        # df = pd.DataFrame({"Wealth": wealth_index, 
        #                     "Previous Peak": previous_peaks, 
        #                     "Drawdown": drawdowns})
        return drawdowns

    def skewness(self):
        """
        Alternative to scipy.stats.skew()
        Computes the skewness of the supplied Series or DataFrame
        Returns a float or a Series
        """
        r = self.returns
        demeaned_r = r - r.mean()
        # use the population standard deviation, so set dof=0
        sigma_r = r.std(ddof=0)
        exp = (demeaned_r**3).mean()
        return exp/sigma_r**3

    def kurtosis(self):
        r = self.returns
        """
        Alternative to scipy.stats.kurtosis()
        Computes the skewness of the supplied Series or DataFrame
        Returns a float or a Series
        """
        demeaned_r = r - r.mean()
        # use the population standard deviation, so set dof=0
        sigma_r = r.std(ddof=0)
        exp = (demeaned_r**4).mean()
        return exp/sigma_r**4

    def is_normal(self, level=0.01):
        r = self.returns
        """
        Applies the Jarque-Bera test to determine if a Series is normal or not
        Test is applied at the 1% level by default
        Returns True if the hypothesis of normality is accepted, False otherwise
        """
        test = r.aggregate(scipy.stats.jarque_bera)
        test = 0.05<test.iloc[1]
        return test

    def semideviation(self):
        r = self.returns
        """
        Returns the semideviation aka negative semideviation of r
        r must be a Series or a DataFrame, else raises a TypeError
        """
        excess= r-r.mean()                                        # We demean the returns
        excess_negative = excess[excess<0]                        # We take only the returns below the mean
        excess_negative_square = excess_negative**2               # We square the demeaned returns below the mean
        n_negative = len(excess_negative)                        # number of returns under the mean
        return (excess_negative_square.sum()/n_negative)**0.5 

    def VaR_historic(self, level=5):
        r = self.returns
        """
        Returns the historic Value at Risk at a specified level
        i.e. returns the number such that "level" percent of the returns
        fall below that number, and the (100-level) percent are above
        """
        vars = r.aggregate(np.percentile, q=level)
        return vars
        
        # else:
        #     raise TypeError("Expected r to be a Series or DataFrame") 

    def CVaR_historic(self, level=5):
        r = self.returns
        """
        Returns the historic Value at Risk at a specified level
        i.e. returns the number such that "level" percent of the returns
        fall below that number, and the (100-level) percent are above
        """

        def single_serie(ser):
            var = np.percentile(ser, q=level)
            is_beyond = ser <= -var
            return -ser[is_beyond].mean()
        
        cvars = r.aggregate(single_serie)
        return cvars
        
    def VaR_gaussian(self, level=5, modified=False):
        r = self.returns
        """
        Returns the Parametric Gauusian VaR of a Series or DataFrame
        If "modified" is True, then the modified VaR is returned,
        using the Cornish-Fisher modification
        """
        # compute the Z score assuming it was Gaussian
        z = norm.ppf(level/100)
        if modified:
            # modify the Z score based on observed skewness and kurtosis
            s = self.skewness()
            k = self.kurtosis()
            z = (z +
                    (z**2 - 1)*s/6 +
                    (z**3 -3*z)*(k-3)/24 -
                    (2*z**3 - 5*z)*(s**2)/36
                )
            
        return -(r.mean() + z*r.std(ddof=0))

    def sharpe_ratio(self, riskfree_rate, periods_per_year=12):
        r = self.returns
        """
        Computes the annualized sharpe ratio of a set of returns
        """
        # convert the annual riskfree rate to per period
        rf_per_period = (1+riskfree_rate)**(1/periods_per_year)-1
        excess_ret = r - rf_per_period
        ann_ex_ret = self.annualize_rets(excess_ret, periods_per_year)
        ann_vol = self.annualize_vol(r, periods_per_year)
        return ann_ex_ret/ann_vol

class MPT:
    @staticmethod
    def portfolio_return(weights, returns):
        """
        Computes the return on a portfolio from constituent returns and weights
        weights are a numpy array or Nx1 matrix and returns are a numpy array or Nx1 matrix
        """
        return weights.T @ returns
    @staticmethod
    def portfolio_vol(weights, covmat):
        """
        Computes the vol of a portfolio from a covariance matrix and constituent weights
        weights are a numpy array or N x 1 maxtrix and covmat is an N x N matrix
        """
        return (weights.T @ covmat @ weights)**0.5
    @staticmethod
    def plot_ef2(n_points, er, cov):
        """
        Plots the 2-asset efficient frontier
        """
        if er.shape[0] != 2 or er.shape[0] != 2:
            raise ValueError("plot_ef2 can only plot 2-asset frontiers")
        weights = [np.array([w, 1-w]) for w in np.linspace(0, 1, n_points)]
        rets = [MPT.portfolio_return(w, er) for w in weights]
        vols = [MPT.portfolio_vol(w, cov) for w in weights]
        ef = pd.DataFrame({
            "Returns": rets, 
            "Volatility": vols
        })
        return ef.plot.line(x="Volatility", y="Returns", style=".-")
    @staticmethod
    def plot_ef(er, cov, n_points=30):
        """
        Plots the multi-asset efficient frontier
        """
        weights = MPT.optimal_weights(n_points, er, cov) # find the allocation of weights in the given of return
        rets = [MPT.portfolio_return(w, er) for w in weights]
        vols = [MPT.portfolio_vol(w, cov) for w in weights]
        ef = pd.DataFrame({
            "Returns": rets, 
            "Volatility": vols
        })
        return ef.plot.line(x="Volatility", y="Returns", style='.-')
    @staticmethod
    def optimal_weights(n_points, er, cov):
        """
        """
        target_rs = np.linspace(er.min(), er.max(), n_points)
        weights = [MPT.minimize_vol(target_return, er, cov) for target_return in target_rs]
        return weights
    @staticmethod
    def minimize_vol(target_return, er, cov):
        """
        Returns the optimal weights that achieve the target return
        given a set of expected returns and a covariance matrix
        """
        n = er.shape[0]
        init_guess = np.repeat(1/n, n)
        bounds = ((0.0, 1.0),) * n # * n applies to every asset

        # construct the constraints
        #   Function needs to be zero
        weights_sum_to_1 = {'type': 'eq',
                            'fun': lambda weights: np.sum(weights) - 1
        }
        return_is_target = {'type': 'eq',
                            'args': (er,),
                            'fun': lambda weights, er: target_return - MPT.portfolio_return(weights,er)
        }
        weights = minimize(MPT.portfolio_vol, init_guess,
                        args=(cov,), method='SLSQP',
                        options={'disp': False},
                        constraints=(weights_sum_to_1,return_is_target),
                        bounds=bounds)
        return weights.x
    @staticmethod
    def msr(riskfree_rate, er, cov):
        """
        Returns the weights of the portfolio that gives you the maximum sharpe ratio
        given the riskfree rate and expected returns and a covariance matrix
        """
        n = er.shape[0]
        init_guess = np.repeat(1/n, n)
        bounds = ((0.0, 1.0),) * n # an N-tuple of 2-tuples!
        # construct the constraints
        weights_sum_to_1 = {'type': 'eq',
                            'fun': lambda weights: np.sum(weights) - 1
        }
        def neg_sharpe(weights, riskfree_rate, er, cov):
            """
            Returns the negative of the sharpe ratio
            of the given portfolio

            Objective function
            """
            r = MPT.portfolio_return(weights, er)
            vol = MPT.portfolio_vol(weights, cov)
            return -(r - riskfree_rate)/vol
        
        weights = minimize(neg_sharpe, init_guess,
                        args=(riskfree_rate, er, cov), method='SLSQP',
                        options={'disp': False},
                        constraints=(weights_sum_to_1,),
                        bounds=bounds)
        return weights.x
    @staticmethod
    def gmv(cov):
        """
        Returns the weights of the Global Minimum Volatility portfolio
        given a covariance matrix
        """
        n = cov.shape[0]
        return MPT.msr(0, np.repeat(1, n), cov)
    # expected returnleri ayni verdigim icin obj fonksiyonunu tek etkileyen sey volatility oluyor, amac volatility i minimize etmek
    @staticmethod
    def plot_ef(n_points, er, cov, show_cml=True, riskfree_rate=0, show_ew=True, show_gmv=True):
        """
        Plots the multi-asset efficient frontier
        """
        weights = MPT.optimal_weights(n_points, er, cov) # find the allocation of weights in the given of return
        rets = [MPT.portfolio_return(w, er) for w in weights]
        vols = [MPT.portfolio_vol(w, cov) for w in weights]
        ef = pd.DataFrame({
            "Returns": rets, 
            "Volatility": vols
        })
        fig, ax = plt.subplots(figsize=(10,5))
        
        
        # ax = ef.plot.line(x="Volatility", y="Returns", style='.-')
        ax.scatter(x=ef["Volatility"], y=ef["Returns"], marker ='.',)

        if show_cml:
            ax.set_xlim(left = 0)
            # get MSR
            w_msr = MPT.msr(riskfree_rate, er, cov)
            r_msr = MPT.portfolio_return(w_msr, er)
            vol_msr = MPT.portfolio_vol(w_msr, cov)
            # add CML
            cml_x = [0, vol_msr]
            cml_y = [riskfree_rate, r_msr]
            ax.plot(cml_x, cml_y, color='green', marker='o', linestyle='dashed', linewidth=2, markersize=12)
        
        if show_ew:
            n = er.shape[0]
            w_ew = np.repeat(1/n, n)
            r_ew = MPT.portfolio_return(w_ew, er)
            vol_ew = MPT.portfolio_vol(w_ew, cov)
            # add EW
            ax.plot([vol_ew], [r_ew], color='goldenrod', marker='o', markersize=10)

        if show_gmv:
            w_gmv = MPT.gmv(cov)
            r_gmv = MPT.portfolio_return(w_gmv, er)
            vol_gmv = MPT.portfolio_vol(w_gmv, cov)
            # add EW
            ax.plot([vol_gmv], [r_gmv], color='midnightblue', marker='o', markersize=10)
        
        return fig