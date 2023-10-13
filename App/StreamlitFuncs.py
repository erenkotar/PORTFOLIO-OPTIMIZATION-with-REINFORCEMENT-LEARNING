import streamlit as st

def about():
    expander = st.sidebar.expander('**About**')
    expander.write("This tool is prepared by [👨‍💻 Eren KOTAR](https://linktr.ee/erenkotar) for the  \
                graduation [thesis](https://www.researchgate.net/publication/374069161_PORTFOLIO_OPTIMIZATION_with_REINFORCEMENT_LEARNING)")
    expander.write("@ Istanbul Technical University, Department of Industrial Engineering, 2023")
    expander.write("Feel free to share your feedback :smiley:")
    expander.write("*Last update: 2023-10-13*")