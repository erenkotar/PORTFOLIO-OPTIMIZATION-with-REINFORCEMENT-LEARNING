import streamlit as st

# def main():
#     # Check if the session state already has a portfolio
#     if 'portfolio' not in st.session_state:
#         st.session_state.portfolio = []

#     # Create the multiselect widget for stock selection
#     selected_stocks = st.multiselect("Select Stocks for your Portfolio:", ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"], st.session_state.portfolio)

#     # Create a submit button and save selected stocks to session state when clicked
#     if st.button("Submit"):
#         st.session_state.portfolio = selected_stocks
#         st.write("Your portfolio has been updated!")

#     # Display the current portfolio
#     st.write("Your Current Portfolio:")
#     for stock in st.session_state.portfolio:
#         st.write(stock)

#     # Create a reset button to clear the portfolio
#     if st.button("Reset Portfolio"):
#         st.session_state.portfolio = []
#         st.write("Your portfolio has been reset!")

# if __name__ == "__main__":
#     main()
