from finance_module import *

write_align(text="Markowitz Optimization", pos="left", style="h1")

if 'fin_obj' in st.session_state:
    if st.button("OPTIMIZE"):
        fin_obj = st.session_state["fin_obj"]
        covv = fin_obj.returns.cov()
        err = fin_obj.annualize_rets(r=fin_obj.returns, periods_per_year=12)
        fig=MPT.plot_ef(n_points=30, er=err, cov=covv, show_cml=False, riskfree_rate=0, show_ew=True, show_gmv=True)
        st.pyplot(fig=fig,)
else:
    st.info("Construct a portfolio first at **Inputs Page!**")
    st.stop()
