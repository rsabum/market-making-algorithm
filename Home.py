import streamlit as st
st.set_page_config(layout="wide")


st.title("High Frequency Market Making")
st.write('This project implements a high-frequency market-making algorithm based on the Avellaneda-Stoikov framework. The goal is to optimize bid-ask quotes in real time to maximize profit while managing risk from holding inventory. Using stochastic control techniques, we derive and numerically solve a Hamilton-Jacobi-Bellman Quasi-Variational Inequality (HJB-QVI) that governs optimal trading decisions. The resulting strategy balances aggressive quoting with careful inventory control. Through Monte Carlo simulations, we show the strategy consistently yields profit with high probability, highlighting its potential for real-world application.')



