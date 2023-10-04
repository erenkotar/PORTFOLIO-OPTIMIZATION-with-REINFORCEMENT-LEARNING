from finance_module import *

write_align(text="Visualization / Simple Anlaysis", pos="left", style="h1")
st.caption("In this page you can visualize the stocks' prices, returns and conduct some simple analysis.")
st.caption("Try filtering the stocks', changing the period!")

if 'fin_obj' in st.session_state:
    fin_obj = st.session_state['fin_obj']
    with st.spinner("Please wait for visualizations..."):
        price_fig = fin_obj.scatter_graph_price()
        st.plotly_chart(price_fig)
        
        price_fig = fin_obj.scatter_graph_cumreturn()
        st.plotly_chart(price_fig)
        
        return_fig = fin_obj.scatter_graph_return()
        st.plotly_chart(return_fig)

    # show_candle = st.radio("Candle Stick Graph of All Stocks", 
    #                     options=("Show", "Don't show"), key="show_candle")
    # if show_candle=="Show":
    #     candles_fig = fin_obj.candlestick_graph()
    #     st.plotly_chart(candles_fig)
    
        historgram_fig = fin_obj.histogram_graph()
        st.plotly_chart(historgram_fig)

    st.title("Summary Report")
    show_report = st.radio("Return Report", 
                        options=("Don't show", "Show"), key="show_report")
    
    if show_report=="Show":
    
        select_period = st.number_input('Return Period in Month',
                            value = 12,
                            min_value=1,
                            max_value=12)
        
        select_rfree_rate = st.number_input('Risk-Free Rate (default for Turkey):',
                            value=0.3,
                            min_value=0.0001,
                            max_value=0.5)
        
        select_pvalue = st.number_input('P-value that will be used in hypotesis testing:',
                            value=0.05,
                            min_value=0.01,
                            max_value=0.15)
        
        if st.button("SUBMIT PARAMETERS"):
            report_df = fin_obj.conduct_all_analysis(period=select_period, level=select_pvalue, rfree_rate=select_rfree_rate, modified=False)
            st.table(report_df)
else:
    st.info("Construct a portfolio first at **Inputs Page**!")
    st.stop()