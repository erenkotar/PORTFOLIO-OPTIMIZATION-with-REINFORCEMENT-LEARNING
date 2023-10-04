from thesis_module import *

write_align(text="Visualization / Simple Anlaysis", pos="left", style="h1")

if 'selected_stocks' in st.session_state:
    tickers_obj = Finance(st.session_state["selected_stocks"])
    st.session_state["tickers_obj"] = tickers_obj


    show_line = st.radio("Stocks' Line Chart", 
                        options=("Show", "Don't show"), key="show_line")
    if show_line=="Show":
        line_fig = tickers_obj.line_graph()
        st.plotly_chart(line_fig)

    show_candle = st.radio("Candle Stick Graph of All Stocks", 
                        options=("Show", "Don't show"), key="show_candle")
    if show_candle=="Show":
        candles_fig = tickers_obj.candlestick_graph()
        st.plotly_chart(candles_fig)

    show_historgram = st.radio("Histogram Graph of All Stocks' Returns", 
                        options=("Show", "Don't show"), key="show_historgram")
    if show_historgram=="Show":
        historgram_fig = tickers_obj.histogram_graph()
        st.plotly_chart(historgram_fig)

    show_report = st.radio("Return Report", 
                        options=("Show", "Don't show"), key="show_report")
    if show_candle=="Show":
        select_period = st.number_input('Return Period in Month (default: 12):',
                            min_value=1,
                            max_value=12)
        
        select_rfree_rate = st.number_input('Risk-Free Rate (in Annualized unit):',
                            min_value=0.0001,
                            max_value=0.20)
        
        select_pvalue = st.select_slider('P-value that will be used in hypotesis testing:',
                            options=[0.01,0.05,0.1])
        
        if st.button("SUBMIT PARAMETERS"):
            report_df = tickers_obj.conduct_all_analysis(period=select_period, level=select_pvalue, rfree_rate=select_rfree_rate, modified=False)
            st.table(report_df)