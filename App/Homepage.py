from finance_module import *

tabs= ["Welcome","Inputs","Visualization / Simple Anlaysis", "Portfolio Optimization", "About"]
st.sidebar.success("Select a page above.")

st.markdown("<h1 style='text-align:center;'>Stock Anlaysis Tool</h1>",unsafe_allow_html=True)
st.write("""\nThis tool is allows you to: """)
st.write("""    - Conduct Analysis on selected stocks with some visualization options and common investment metrics""")
st.write("""    - Managing the allocation of those stocks with different efficient fronteirs""")