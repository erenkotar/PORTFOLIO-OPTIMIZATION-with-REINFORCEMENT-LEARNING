from finance_module import *
from StreamlitFuncs import *

about()

# tabs= ["Welcome","Inputs","Visualization / Simple Anlaysis", "Portfolio Optimization", "About"]
st.sidebar.success("Select a page above.")

st.title("Stock Analysis Tool")

st.image("app/stocks2.jpg")  # Change to your desired image, this is just a placeholder

st.write("""
### Welcome to the Stock Analysis Tool!
This tool is designed to provide you analyzing stocks and optimizing your investment strategy.
""")

st.write("""
    ## Features:
    - **Stock Analysis**: Dive deep into the performance of selected stocks with interactive visualizations and gain insights from common investment metrics.
    - **Portfolio Management**: Manage your stock allocations, evaluate potential risks, and find the most efficient investment frontiers for optimal returns.
    """)

st.write("""
## Instructions:
1. **Select your Stocks**: Browse and select from a comprehensive list of available stocks.
2. **Analyze**: View interactive charts, metrics, and historical data to evaluate stock performance.
3. **Optimize Portfolio**: Input your desired allocations and visualize potential returns and risks on the efficient frontier.
""")


