import numpy as np
import streamlit as st
st.set_page_config(layout="wide")

st.title("Model")

st.markdown("### Stochastic Control Formulation")
st.markdown(r"""
    We model the market maker's task as a stochastic control problem: optimize bid and ask quote depths to **maximize expected profit** by trading the spread, while **minimizing risk** from holding inventory. Two penalties are applied:

    - **Running Inventory Penalty**: discourages large positions during the session.
    - **Terminal Inventory Penalty**: discourages ending the session with inventory.

    The objective is to choose quotes that maximize expected terminal wealth while accounting for these penalties. This leads to a the following objective functthat captures the essence of this problem:
    $$
    V(t, s_{t}, x_{t}, q_{t}) = \max_{\delta^{\text{b}}, \delta^{\text{a}}}~ \mathbb{E} 
            \left[ x_{T} + q_{T}(s_{T} - \alpha q_{T}) - \phi \int_{t}^{T} q_{s}^{2} \, ds \right]
    $$
    Where:
    - $s_{t}$ is the mid-price of the asset at time $t$  
    - $x_{t}$ is the market maker's cash at time $t$  
    - $q_{t}$ is the market maker's inventory at time $t$  
    - $\delta^{\text{b}}$ is how deep in the market we quote our bid price  
    - $\delta^{\text{a}}$ is how deep in the market we quote our ask price  
    - $\alpha > 0$ is the terminal inventory penalty parameter  
    - $\phi > 0$ is the running inventory penalty parameter
""")

st.markdown('---')
st.markdown("### Market Dynamics")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(r"""
        ##### Reserve-Price -- $s_t$

        The reserve-price of the market follows a Bachelier process:
        $$
        ds_t = \sigma \, dW_t
        $$
        where $\sigma$ is the volatility and $W_t$ is a standard Brownian motion increment.
    """)

with col2:
    st.markdown(r"""
        ##### Order Flow -- $\lambda(t, s, \delta)$

        Market order arrivals follow Poisson processes with intensities decreasing in quote depth:
        $$
        \begin{align*}
        \lambda^{\text{b}}(t, s, \delta^{\text{b}}) &= \Lambda^{\text{b}} e^{-\kappa^{\text{b}} \delta^{\text{b}}} \\
        \lambda^{\text{a}}(t, s, \delta^{\text{a}}) &= \Lambda^{\text{a}} e^{-\kappa^{\text{a}} \delta^{\text{a}}}
        \end{align*}
        $$
    """)

# col3, col4 = st.columns(2)
with col3:
    st.markdown(r"""
        ##### Inventory -- $q_t$

        Inventory changes with incoming orders:
        $$
        dq_t = dN_t^{\text{b}} - dN_t^{\text{a}}
        $$
        where $dN_t^{\text{b}}$ and $dN_t^{\text{a}}$ represent the incoming buy and sell orders at time $t$ respectively.
    """)

with col4:
    st.markdown(r"""
        ##### Cash -- $x_t$

        Cash evolves from filled orders and rebates:
        $$
        \begin{align*}
        dx_t &= (s_t + \delta^{\text{a}} + \epsilon)\, dN_t^{\text{a}} \\
             &- (s_t - \delta^{\text{b}} + \epsilon)\, dN_t^{\text{b}}
        \end{align*}
        $$
        where $\epsilon$ is the rebate awarded every time an order hits the market maker's quote.
    """)

st.markdown('---')
st.markdown("### Optimal Trading Strategy")
st.write(r"""
    ##### Hamilton-Jacobi-Bellman Equation
    
    To determine the optimal trading strategy, we take a stochastic control approach and solve the following Hamilton-Jacobi-Bellman equation:
    $$
    \begin{align*}
        0 & = \partial_{t}V + \frac{1}{2}\partial_{ss}V \sigma^{2} \\
          & + \underset{\delta^\text{b}}{\max}\big( \Lambda^{\text{b}}e^{-\kappa^{\text{b}} \delta^\text{b}}
            \big[ V(t, x-(s-\delta^{\text{b}})+ \epsilon, s, q+1) - V(t, x, s, q) \big] \big) \\
          & + \underset{\delta^\text{a}}{\max}\big( \Lambda^{\text{a}}e^{-\kappa^{\text{a}} \delta^\text{a}}
            \big[ V(t, x+(s+\delta^{\text{a}})+ \epsilon, s, q-1) - V(t, x, s, q) \big] \big) \\
          & - \phi q^{2}
    \end{align*}
    $$
""")  


with st.expander("Derivation"):
    st.markdown(r"""
        Let's recall the definition of the infinitesimal generator:

        $$
        \begin{align*}
            \mathcal{L}V(t, x, s, q) = \lim_{dt \to 0}\frac{\mathbb{E}\big[ \partial V(t, x, s, q) \big] }{dt}
        \end{align*}
        $$

        We start by applying Ito's lemma to $V(t, x, s, q)$:
        $$
        \begin{align*}
            \partial V(t, x, s, q) & = \partial_{t}V dt + \partial_{s}V dS_{t}+ \frac{1}{2}\partial_{ss}V dS_{t}^{2} \\
                                   & + \big[V(t, x-(s-\delta^{\text{b}}) + \epsilon, s, q+1) - V(t, x, s, q) \big]dN_{t}^{\text{b}} \\
                                   & + \big[V(t, x+(s+\delta^{\text{a}}) + \epsilon, s, q-1) - V(t, x, s, q) \big]dN_{t}^{\text{a}}
        \end{align*}
        $$

        Let's substitute in our model for the spot price
        $dS_{t}= \sigma dW_{t}$ and $dS_{t}^{2}= \sigma^{2}dt$.
        $$
        \begin{align*}
            \partial V(t, x, s, q) & = \partial_{t}V dt + \partial_{s}V (\sigma dW_{t}) + \frac{1}{2}\partial_{ss}V \sigma^{2}dt \\
                                   & + \big[V(t, x-(s-\delta^{\text{b}}) + \epsilon, s, q+1) - V(t, x, s, q) \big]dN_{t}^{\text{b}}       \\
                                   & + \big[V(t, x+(s+\delta^{\text{a}}) + \epsilon, s, q-1) - V(t, x, s, q) \big]dN_{t}^{\text{a}}
        \end{align*}
        $$

        Finally, taking expectation and dividing by $dt$ yields:
        $$
        \begin{align*}
            \mathcal{L}V(t, x, s, q) & = \partial_{t}V + \frac{1}{2}\partial_{ss}V \sigma^{2}                         \\
                                     & + \lambda^{\text{b}}(t, s, \delta^b)\big[V(t, x-(s-\delta^{\text{b}})+ \epsilon, s, q+1) - V(t, x, s, q) \big] \\
                                     & + \lambda^{\text{a}}(t, s, \delta^a)\big[V(t, x+(s+\delta^{\text{a}})+ \epsilon, s, q-1) - V(t, x, s, q) \big]
        \end{align*}
        $$

        Since we made the assumption that
        $\lambda(t, s, \delta)= \Lambda e^{-\kappa \delta}$ we get that:
        $$
        \begin{align*}
            \mathcal{L}V(t, x, s, q) & = \partial_{t}V + \frac{1}{2}\partial_{ss}V \sigma^{2} \\
                                     & + \Lambda^{\text{b}}e^{-\kappa^{\text{b}} \delta^\text{b}}
                                        \big[V(t, x-(s-\delta^{\text{b}})+ \epsilon, s, q+1) - V(t, x, s, q) \big] \\
                                     & + \Lambda^{\text{a}}e^{-\kappa^{\text{a}} \delta^\text{a}}
                                        \big[V(t, x+(s+\delta^{\text{a}})+ \epsilon, s, q-1) - V(t, x, s, q) \big]
        \end{align*}
        $$
                
        Therefore our Hamilton Jacobi Bellman is:
        $$
        \begin{align*}
            0 & = \partial_{t}V + \frac{1}{2}\partial_{ss}V \sigma^{2} \\
              & + \underset{\delta^\text{b}}{\max}\big( \Lambda^{\text{b}}e^{-\kappa^{\text{b}} \delta^\text{b}}
                    \big[ V(t, x-(s-\delta^{\text{b}})+ \epsilon, s, q+1) - V(t, x, s, q) \big] \big) \\
              & + \underset{\delta^\text{a}}{\max}\big( \Lambda^{\text{a}}e^{-\kappa^{\text{a}} \delta^\text{a}}
                    \big[ V(t, x+(s+\delta^{\text{a}})+ \epsilon, s, q-1) - V(t, x, s, q) \big] \big) \\
              & - \phi q^{2}
        \end{align*}
        $$
    """)
     
st.write(r"""
    ##### Linear Ansatz
         
    We can simplify the PDE by assuming its solution takes the linear form 
    $$V(t, s, x, q) = x + sq + v(t, q)$$
    substituting this back in yields:
    $$
    \begin{align*}
        0 & = \partial_{t}v - \phi q^{2} \\
          & + \underset{\delta^\text{b}}{\max}\big( \Lambda^{\text{b}}e^{-\kappa^{\text{b}} \delta^\text{b}}
                \big[ \delta^{\text{b}}+ \epsilon + v(t, q+1) - v(t, q) \big] \big) \\
          & + \underset{\delta^\text{a}}{\max}\big( \Lambda^{\text{a}}e^{-\kappa^{\text{a}} \delta^\text{a}}
                \big[ \delta^{\text{a}}+ \epsilon + v(t, q-1) - v(t, q) \big] \big)
    \end{align*}
    $$
    with the boundary condition $v(T, q_{T}) = -\alpha q_{T}^{2}$.
""")  

st.markdown(r"""
    ##### Optimal Bid & Ask Depths

    At each time step, the optimal bid and ask depths solve:
    $$
    \delta^b = \max\left(0,\ \frac{1}{\kappa^b} - \epsilon - v(t, q + 1) + v(t, q)\right)
    $$
    $$
    \delta^a = \max\left(0,\ \frac{1}{\kappa^a} - \epsilon - v(t, q - 1) + v(t, q)\right)
    $$
    These control how far the market maker posts quotes from the mid-price, balancing profitability and risk.
""")
with st.expander("Derivation"):
    st.markdown(r"""
        Let's recall the definition of the infinitesimal generator:

        $$
        \begin{align*}
            \mathcal{L}V(t, x, s, q) = \lim_{dt \to 0}\frac{\mathbb{E}\big[ \partial V(t, x, s, q) \big] }{dt}
        \end{align*}
        $$

        We start by applying Ito's lemma to $V(t, x, s, q)$:
        $$
        \begin{align*}
            \partial V(t, x, s, q) & = \partial_{t}V dt + \partial_{s}V dS_{t}+ \frac{1}{2}\partial_{ss}V dS_{t}^{2} \\
                                   & + \big[V(t, x-(s-\delta^{\text{b}}) + \epsilon, s, q+1) - V(t, x, s, q) \big]dN_{t}^{\text{b}} \\
                                   & + \big[V(t, x+(s+\delta^{\text{a}}) + \epsilon, s, q-1) - V(t, x, s, q) \big]dN_{t}^{\text{a}}
        \end{align*}
        $$

        Since we assume Bachelier type model for the spot price, we can substitute
        $dS_{t}= \mu dt + \sigma dW_{t}$ and $dS_{t}^{2}= \sigma^{2}dt$.
        $$
        \begin{align*}
            \partial V(t, x, s, q) & = \partial_{t}V dt + \partial_{s}V (\mu dt + \sigma dW_{t}) + \frac{1}{2}\partial_{ss}V \sigma^{2}dt \\
                                   & + \big[V(t, x-(s-\delta^{\text{b}}) + \epsilon, s, q+1) - V(t, x, s, q) \big]dN_{t}^{\text{b}}       \\
                                   & + \big[V(t, x+(s+\delta^{\text{a}}) + \epsilon, s, q-1) - V(t, x, s, q) \big]dN_{t}^{\text{a}}
        \end{align*}
        $$
        Finally, taking expectation and dividing by $dt$ yields:
        $$
        \begin{align*}
            \mathcal{L}V(t, x, s, q) & = \partial_{t}V + \frac{1}{2}\partial_{ss}V \sigma^{2}                         \\
                                     & + \Lambda_{t}^{\text{b}}\big[V(t, x-(s-\delta^{\text{b}})+ \epsilon, s, q+1) - V(t, x, s, q) \big] \\
                                     & + \Lambda_{t}^{\text{a}}\big[V(t, x+(s+\delta^{\text{a}})+ \epsilon, s, q-1) - V(t, x, s, q) \big]
        \end{align*}
        $$
        Since we made the explicit assumption that
        $\Lambda_{t}= \Lambda e^{-\kappa \delta}$ we get that:
        $$
        \begin{align*}
            \mathcal{L}V(t, x, s, q) & = \partial_{t}V + \frac{1}{2}\partial_{ss}V \sigma^{2} \\
                                     & + \Lambda^{\text{b}}e^{-\kappa^{\text{b}} \delta^\text{b}}
                                        \big[V(t, x-(s-\delta^{\text{b}})+ \epsilon, s, q+1) - V(t, x, s, q) \big] \\
                                     & + \Lambda^{\text{a}}e^{-\kappa^{\text{a}} \delta^\text{a}}
                                        \big[V(t, x+(s+\delta^{\text{a}})+ \epsilon, s, q-1) - V(t, x, s, q) \big]
        \end{align*}
        $$
                
        Therefore our Hamilton Jacobi Bellman is:
        $$
        \begin{align*}
            0 & = \partial_{t}V + \frac{1}{2}\partial_{ss}V \sigma^{2} \\
              & + \underset{\delta^\text{b}}{\max}\big( \Lambda^{\text{b}}e^{-\kappa^{\text{b}} \delta^\text{b}}
                    \big[ V(t, x-(s-\delta^{\text{b}})+ \epsilon, s, q+1) - V(t, x, s, q) \big] \big) \\
              & + \underset{\delta^\text{a}}{\max}\big( \Lambda^{\text{a}}e^{-\kappa^{\text{a}} \delta^\text{a}}
                    \big[ V(t, x+(s+\delta^{\text{a}})+ \epsilon, s, q-1) - V(t, x, s, q) \big] \big) \\
              & - \phi q^{2}
        \end{align*}
        $$
    """)