import streamlit as st

st.title("About This Dashboard")
st.markdown("""
Dashboard that provides insights into house pricing data. 
        The dashboard is designed to be user-friendly, interactive, and informative, helping users visualize and analyze trends in house prices.
""")

# Connect with Developer Button
st.markdown("""
<style>
.button {
    background-color: yellow;
    border: none;
    color: white;
    padding: 10px 20px;
    text-align: center;
    text-decoration: none;
    display: inline-block;
    font-size: 16px;
    margin: 4px 2px;
    cursor: pointer;
    border-radius: 4px;
    font-weight: bold;
}
</style>
<a href="https://rjscrapy.pythonanywhere.com/developer" target="_blank" class="button">Connect with Developer</a>
""", unsafe_allow_html=True)