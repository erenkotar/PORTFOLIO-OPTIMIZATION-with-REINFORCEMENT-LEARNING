from finance_module import *

write_align(text="Markowitz Optimization", pos="left", style="h1")

if st.button("OPTIMIZE"):
    tickers_obj = st.session_state["tickers_obj"]
    covv = tickers_obj.returns.cov()
    err = tickers_obj.annualize_rets(r=tickers_obj.returns, periods_per_year=12)
    fig=MPT.plot_ef(n_points=30, er=err, cov=covv, show_cml=False, riskfree_rate=0, show_ew=True, show_gmv=True)
    st.pyplot(fig=fig,)