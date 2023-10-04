from thesis_module import *

st.title("Inputs of Portfolio")
st.caption("""Select the stocks you want for portfolio **SAVE** the portfolio!""")
st.caption("""Remember that you can always **RESET** the portfolio and start over!""")

# write_align(text="Inputs of Portfolio", pos="left", style="strong")
# write_align(text="This page is allows you to select the stocks and clear the cache", pos="left")

def initilize():
    st.session_state["p1_buttons"] = {
        "save":False
    }
    st.session_state["initiate"] = True

if "initilize" not in st.session_state:
    initilize()

submit_2_1 = st.button("RESET PORTFOLIO")

if submit_2_1:
    if st.session_state["p1_buttons"]["save"] == True:
        # st.session_state["p1_buttons"]["save"] = False
        del st.session_state["selected_stocks"]
        st.success("All stocks in portfolio successfully deleted!")
        st.experimental_rerun()
    else:
        st.error("You need to save the portfolio first!")

stocks_selected = st.multiselect("**Select all the preffered stocks you want to conduct analysis and optimize the portfolio and click the **SAVE PORTFOLIO** button at the end of the page:**", 
                               options=("AMZN", "APPL", "MSFT","GOOG","META","NVDA","V","PG","NFLX"))


submit_2_2=st.button("SAVE PORTFOLIO")
# try:
#     Finance(stocks_selected)
# except BaseException:
#     raise("Some of the stocks are not reachable at the moment!")
# else:
if submit_2_2:
    st.session_state["selected_stocks"] = stocks_selected
    st.session_state["p1_buttons"]["save"] = True
    st.success("All stocks are successfully saved to memory!")
