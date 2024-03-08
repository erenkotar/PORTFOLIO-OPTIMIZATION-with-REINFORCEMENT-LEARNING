from finance_module import *
from StreamlitFuncs import *

about()

st.title("Inputs of Portfolio")

def initilize():
    st.session_state["p1_buttons"] = {
        # "save":False,
        "reset":False,
        "saved":False

    }
    st.session_state["initilize"] = True

if "initilize" not in st.session_state:
    initilize()

tickers_df = pd.read_parquet("app/data/MetaTickerFil.parquet")

if st.session_state["p1_buttons"]["saved"]:
    st.warning("A portfolio is already selected. You can continue with the current selection or reset to make a new selection.")


selected_stocks = st.multiselect("Select all the preffered stocks you want to conduct analysis and optimize the portfolio and click the **SAVE PORTFOLIO** button at the end of the page:", 
                               options=tickers_df.Ticker.tolist())

start_date = st.date_input("Select the start date:", value=datetime.datetime(2018,1,31))
end_date = st.date_input("Select the end date:", value=datetime.datetime(2023,1,31))

col1, col2 = st.columns(2)

# Put a button in each column
col1.caption("""Click the **SAVE** button after selection!""")
col2.caption("""You can always **RESET** the portfolio and start over!""")

save_button = col1.button("SAVE PORTFOLIO")
reset_button = col2.button("RESET PORTFOLIO")

if reset_button:
    st.session_state["p1_buttons"]["reset"] = True
    for i in st.session_state.keys():
        del st.session_state[i]
    st.experimental_rerun()

if st.session_state["p1_buttons"]["saved"]==False:
    if save_button:
        if (start_date > end_date):
            st.error("Start date cannot be later than end date!")
            st.stop()
        if (len(selected_stocks) < 2):
            st.error("You need to select at least two stocks to construct a portfolio!")
            st.stop()

        with st.spinner("Please wait while we are fetching the data..."):
            try:
                fin_obj = Finance(selected_stocks, start_date, end_date)
            except BaseException:
                raise Exception("Some of the stocks are not reachable at the moment!")

        st.session_state["selected_stocks"] = selected_stocks
        st.session_state["fin_obj"] = fin_obj
        st.session_state["p1_buttons"]["saved"] = True
        st.success("All stocks are successfully saved to memory!")
    else:
        st.stop()

if st.session_state["p1_buttons"]["saved"]:
    # st.write("Your Current Portfolio:")
    all_data, daily_data, news_data = st.tabs(["All Data", "Price Data", "News Data"])
    
    with all_data:
        temp_data = st.session_state["fin_obj"].all_data.copy()
        temp_data.columns = [i[1]+"_"+i[0] for i in temp_data.columns]
        st.header("All Data")
        st.write(temp_data)

    with daily_data:
        st.header("Price Data")
        st.write(st.session_state["fin_obj"].prices)

    # with fundamental_data:
        # st.write(st.session_state["fin_obj"].fundamental_data)

    if 'news_interaction' not in st.session_state:
        st.session_state['news_interaction'] = False

    with news_data:
        if 'selected_stock' not in st.session_state:
            st.session_state['selected_stock'] = st.session_state["selected_stocks"][0]

        selected_stock = st.radio("Select ticker!", options=st.session_state["selected_stocks"])

        if selected_stock != st.session_state['selected_stock']:
            st.session_state['selected_stock'] = selected_stock
            st.session_state['news_interaction'] = True

        st.write(st.session_state["fin_obj"].get_stock_news())
        # sn = StockNews(selected_stock, save_news=False)
        # news_df = sn.read_rss()
        # st.write(news_df)

    # for stock in st.session_state["selected_stocks"]:
    #     st.write(stock)

# reset_button = st.button("RESET PORTFOLIO", 
#                        disabled=~st.session_state["p1_buttons"]["save"])

# if reset_button:
#     st.write(st.session_state["p1_buttons"]["save"])
#     if st.session_state["p1_buttons"]["save"] == True:
#         # st.session_state["p1_buttons"]["save"] = False
#         del st.session_state["selected_stocks"]
#         st.success("All stocks in portfolio successfully deleted!")
#         st.experimental_rerun()
#     else:
        # st.error("You need to save the portfolio first!")