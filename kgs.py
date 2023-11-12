import pandas as pd
import plotly.express as px
import streamlit as st
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

st.markdown("<h3 style='text-align: center;'>Chak's Family Kittens Growth Journey</h3>", unsafe_allow_html=True)
st.markdown("")

# URL of the CSV file on Google Drive
url = 'https://drive.google.com/uc?id=1gPtAzNUpH8qxdQnjw5GQknzwTdETJglZ'

# Read the CSV file from the URL
kw = pd.read_csv(url)

kitten_columns = ['A (Male)', 'B (Male)', 'C (Female)', 'D (Female)', 'E (Female)']

# Calculate the mean for each day
kw['Mean'] = kw[kitten_columns].mean(axis=1).round(1)

# Create the line chart using Plotly
fig = px.line(kw, x='Day', y=kitten_columns, title='Gucci & Mui Kitten Growth Record')
fig.add_scatter(x=kw['Day'], y=kw['Mean'], mode='lines', name='Mean', line=dict(color='lightgrey', dash='dash'))

# Set the y-axis range
fig.update_yaxes(range=[75, kw[kitten_columns].values.max()])

# Display the chart using Streamlit
st.plotly_chart(fig)

# Display the data table using Streamlit
st.table(kw)

st.markdown("<h4>OLS Regression Results:</h4>", unsafe_allow_html=True)

# Define the predictors (X) and the target variable (y)
X = kw[['Day']]
y = kw['Mean']

# Create and fit the linear regression model
model = LinearRegression()
model.fit(X, y)

# Add constant to the predictors for statsmodels
X = sm.add_constant(X)

# Create and fit the OLS model
ols_model = sm.OLS(y, X)
results = ols_model.fit()

# Display the OLS regression results using Streamlit
st.text(results.summary())

# Get the regression equation coefficients
slope = round(model.coef_[0], 1)
intercept = round(model.intercept_, 1)

# Display the regression equation using Streamlit
st.markdown("<h4>Regression Equation:</h4>", unsafe_allow_html=True)
st.markdown(f"<pre>Y = {slope} * X + {intercept}</pre>", unsafe_allow_html=True)

# Predict weight for a specific day
future_day = st.number_input("Enter the desired day to predict weight:", value=0, step=1)
future_weight = model.predict([[future_day]])
rounded_weight = round(future_weight[0], 1)
st.markdown(f"<h4>Predicted weight for day {future_day}:</h4>", unsafe_allow_html=True)
st.markdown(f"<pre>{rounded_weight}</pre>", unsafe_allow_html=True)
