from finance_module import *
from StreamlitFuncs import *

about()

write_align(text="Optimization of Portfolio", pos="left", style="h1")

if 'fin_obj' in st.session_state:
    if st.button("OPTIMIZE"):
        fin_obj = st.session_state["fin_obj"]
        covv = fin_obj.returns.cov()
        err = fin_obj.annualize_rets(r=fin_obj.returns, periods_per_year=12)
        fig, weights_df = MPT.plot_ef(n_points=30, er=err, cov=covv, riskfree_rate=0.0)
        st.pyplot(fig=fig,)
        st.write("""
                #### Optimal weights of different methods:
                """)
        st.write(weights_df)
else:
    st.info("Construct a portfolio first at **Inputs Page!**")
    st.stop()
