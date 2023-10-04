from finance_module import *

write_align(text="Visualization / Simple Anlaysis", pos="left", style="h1")
st.caption("In this page you can visualize the stocks' prices, returns and conduct some simple analysis.")
st.caption("Try filtering the stocks', changing the period!")

if 'fin_obj' in st.session_state:
    fin_obj = st.session_state['fin_obj']

    show_price = st.radio("Stocks' Line Chart", 
                        options=("Show", "Don't show"), key="show_price")
    if show_price=="Show":
        price_fig = fin_obj.scatter_graph_price()
        st.plotly_chart(price_fig)
    
    show_price = st.radio("Stocks' Line Chart", 
                        options=("Show", "Don't show"), key="show_price")
    if show_price=="Show":
        price_fig = fin_obj.scatter_graph_price()
        st.plotly_chart(price_fig)
    
    show_return = st.radio("Stocks' Line Chart", 
                        options=("Show", "Don't show"), key="show_return")
    if show_return=="Show":
        return_fig = fin_obj.scatter_graph_return()
        st.plotly_chart(return_fig)

    # show_candle = st.radio("Candle Stick Graph of All Stocks", 
    #                     options=("Show", "Don't show"), key="show_candle")
    # if show_candle=="Show":
    #     candles_fig = fin_obj.candlestick_graph()
    #     st.plotly_chart(candles_fig)

    show_historgram = st.radio("Histogram Graph of All Stocks' Returns", 
                        options=("Show", "Don't show"), key="show_historgram")
    if show_historgram=="Show":
        historgram_fig = fin_obj.histogram_graph()
        st.plotly_chart(historgram_fig)

    show_report = st.radio("Return Report", 
                        options=("Don't show", "Show"), key="show_report")
    select_period = st.number_input('Return Period in Month (default: 12):',
                        min_value=1,
                        max_value=12)
    
    select_rfree_rate = st.number_input('Risk-Free Rate (in Annualized unit):',
                        min_value=0.0001,
                        max_value=0.20)
    
    select_pvalue = st.select_slider('P-value that will be used in hypotesis testing:',
                        options=[0.01,0.05,0.1])
    
    if st.button("SUBMIT PARAMETERS"):
        report_df = fin_obj.conduct_all_analysis(period=select_period, level=select_pvalue, rfree_rate=select_rfree_rate, modified=False)
        st.table(report_df)
else:
    st.info("Construct a portfolio first at **Inputs Page**!")
    st.stop()