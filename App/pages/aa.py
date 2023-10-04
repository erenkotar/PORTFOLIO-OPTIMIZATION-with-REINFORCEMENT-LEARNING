import streamlit as st

col1, col2 = st.beta_columns(2)

# Put a button in each column
save_button = col1.button("SAVE PORTFOLIO")
reset_button = col2.button("RESET")

# Handle button click events
if save_button:
    st.write("You clicked SAVE!")
if reset_button:
    st.write("You clicked RESET!")
    
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
