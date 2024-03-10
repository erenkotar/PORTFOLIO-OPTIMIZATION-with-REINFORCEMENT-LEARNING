from finance_module import *
from StreamlitFuncs import *

about()

write_align(text="Visualizations / Anlaysis", pos="left", style="h1")
st.caption("In this page you can visualize the stocks' prices, returns and conduct some simple analysis.")
st.caption("Try filtering the stocks', changing the period!")

if 'fin_obj' in st.session_state:
    fin_obj = st.session_state['fin_obj']
    with st.spinner("Please wait for visualizations..."):
        price_fig = fin_obj.plot_price()
        st.plotly_chart(price_fig)
        
        cumreturn = fin_obj.plot_cumreturn()
        st.plotly_chart(cumreturn)
        
        return_fig = fin_obj.plot_return()
        st.plotly_chart(return_fig)

        corr_fig = fin_obj.corr_graph()
        st.plotly_chart(corr_fig)

        drawd_fig = fin_obj.drawd_graph()
        st.plotly_chart(drawd_fig)

    # show_candle = st.radio("Candle Stick Graph of All Stocks", 
    #                     options=("Show", "Don't show"), key="show_candle")
    # if show_candle=="Show":
    #     candles_fig = fin_obj.candlestick_graph()
    #     st.plotly_chart(candles_fig)
    
        historgram_fig = fin_obj.histogram_graph()
        st.plotly_chart(historgram_fig)

    st.title("Summary Report")
    # show_report = st.radio("Return Report", 
    #                     options=("Don't show", "Show"), key="show_report")
    
    select_period = st.number_input('Return Period in Days (default: 252):',
                        value = 252,
                        min_value=1,
                        max_value=252)
    
    select_rfree_rate = st.number_input('Risk-Free Rate (default for Turkey):',
                        value=0.3,
                        min_value=0.0001,
                        max_value=0.5)
    
    select_pvalue = st.number_input('P-value that will be used in hypotesis testing:',
                        value=0.05,
                        min_value=0.01,
                        max_value=0.15)

    show_report = st.button("SUBMIT PARAMETERS")

    if show_report:
        with st.spinner("Please wait for report..."):
            report_df = fin_obj.conduct_all_analysis(period=select_period, level=select_pvalue, rfree_rate=select_rfree_rate, modified=False)
            cmap = plt.cm.get_cmap('RdYlGn')
            report_df.style.background_gradient(cmap=cmap,vmin=(-0.015),vmax=0.015,axis=None).format('{:.2%}')
            st.dataframe(report_df.style.background_gradient(cmap=cmap,vmin=(-0.015),vmax=0.015,axis=None))

            fig = fin_obj.report_graph(report_df)
            st.plotly_chart(fig)
    
else:
    st.info("Construct a portfolio first at **Inputs Page!**")
    st.stop()

