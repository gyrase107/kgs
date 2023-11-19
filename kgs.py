import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import streamlit as st
from tabulate import tabulate
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

# SETUP
# URL of the CSV file on Google Drive
url = 'https://github.com/gyrase107/kgs/raw/main/kitten_weight_raw.csv'
kw = pd.read_csv(url)
kitten_columns = ['A (Male)', 'B (Male)', 'C (Female)', 'D (Female)', 'E (Female)']
kw['Mean'] = kw[kitten_columns].mean(axis=1).round(1)

# KA
kwa = kw[['Day', 'A (Male)', 'Mean']]
kwa_col = ['A (Male)']

# KB
kwb = kw[['Day', 'B (Male)', 'Mean']]
kwb_col = ['B (Male)']

# KC
kwc = kw[['Day', 'C (Female)', 'Mean']]
kwc_col = ['C (Female)']

# KD
kwd = kw[['Day', 'D (Female)', 'Mean']]
kwd_col = ['D (Female)']

# KE
kwe = kw[['Day', 'E (Female)', 'Mean']]
kwe_col = ['E (Female)']

# Set default page to 'Homepage'
st.set_page_config(page_title='Homepage')

# Open the sidebar by default
st.sidebar.header('Sidebar - Please Select')

st.sidebar.title("Pages")
selected_page = st.sidebar.selectbox("", ['HomePage', 'Regression Analysis', 'About British Short Hair', '1st Bro', '2nd Bro', '3rd Sis', '4th Sis', '5th Sis', 'Leave Your Comments'])

# HOMEPAGE
if selected_page == 'HomePage':
    
    # Add watermark and copyright statement
    watermark = "Author: Alan Fung"
    copyright = "Copyright © 2023 Alan Fung. All rights reserved."
    
    # Display watermark and copyright statement
    st.markdown(f'<div style="position:absolute; top:10px; right:10px; font-size:12px;">{watermark}</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="position:absolute; bottom:10px; right:10px; font-size:12px;">{copyright}</div>', unsafe_allow_html=True)
    
    st.markdown("<h2 style='text-align: left;'>Chak's Family Kittens Growth Journey</h2>", unsafe_allow_html=True)
    st.markdown("")
    
    st.markdown("<h5>These are my parents, Chak Gucci and Chak Mui:</h5>", unsafe_allow_html=True)
    
    # Display images side by side

    st.image("https://drive.google.com/uc?export=view&id=1B_77o4HNOsl459OfLsze3adg3TohXqnC", width=400)
        
    st.markdown("")
    st.markdown("<h5>Growth Record:</h5>", unsafe_allow_html=True)
    
    # Set the figure size globally
    plt.rcParams['figure.figsize'] = (8, 4)
    plt.rcParams['font.size'] = 10

    # Define colors for each kitten
    colors = ['#FFD700', 'orange', 'red', 'blue', 'green']

    # Create the line chart with grid lines and specified colors
    fig, ax = plt.subplots()
    for i, column in enumerate(kitten_columns):
        kw[column].plot(grid=True, color=colors[i], ax=ax)

    # Calculate the S.D. and G%
    kw['Mean'] = kw[kitten_columns].mean(axis=1).round(1)
    # Add the line for the 'Mean' column

    plt.xlabel('Day No.')
    plt.ylabel('Weight (g)')
    plt.title('Kitten Weight (g) by Day', fontsize=10)
    plt.legend(kitten_columns, fontsize=8)

    # Add the legend for 'Mean' below 'E (Female)'
    handles, labels = plt.gca().get_legend_handles_labels()
    handles.append(plt.Line2D([], [], color='lightgrey', linestyle='--'))
    labels.append('Mean')

    plt.legend(handles, labels, loc='upper left', fontsize=8)

    plt.xlim(0, plt.xlim()[1])  # Set the minimum value of x-axis to 0
    plt.ylim(75, plt.ylim()[1])
    kw['Mean'].plot(grid=True, color='lightgrey', linestyle='--')

    # Set the x-axis tick labels to whole numbers
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # Display the chart using Streamlit
    st.pyplot(fig)
    
    # Calculate the S.D. and G%
    kw['S.D.'] = kw[kitten_columns].std(axis=1).round(1)
    kw['G%'] = (kw['Mean'].pct_change() * 100).fillna(0).round(1)

    new_columns = {
    'A (Male)': 'A',
    'B (Male)': 'B',
    'C (Female)': 'C',
    'D (Female)': 'D',
    'E (Female)': 'E'}
    
    kwt = kw.rename(columns=new_columns)

    # Format the DataFrame column to have one decimal place
    formatted_kwt = kwt.applymap("{:.1f}".format)

    # Convert the formatted DataFrame to a Markdown table
    markdown_table = formatted_kwt.to_markdown(index=False)  # Set index=False to exclude the index column

    # Display the Markdown table with a smaller font size
    st.markdown(markdown_table, unsafe_allow_html=True)  # Add unsafe_allow_html=True to render HTML tags for font size

    
# Regression Analysis      
elif selected_page == 'Regression Analysis':
    
    # Add watermark and copyright statement
    watermark = "Author: Alan Fung"
    copyright = "Copyright © 2023 Alan Fung. All rights reserved."
    
    # Display watermark and copyright statement
    st.markdown(f'<div style="position:absolute; top:10px; right:10px; font-size:12px;">{watermark}</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="position:absolute; bottom:10px; right:10px; font-size:12px;">{copyright}</div>', unsafe_allow_html=True)
    
    st.markdown("<h2 style='text-align: left;'>OLS Regression Result</h2>", unsafe_allow_html=True)
    
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
    
    # Extracting statistical values
    p_value = results.pvalues['Day']
    f_value = round(results.fvalue, 1)
    r_squared = round(results.rsquared, 4)
    adj_r_squared = round(results.rsquared_adj, 4)
    durbin_watson = round(sm.stats.stattools.durbin_watson(results.resid), 3)
    
    # Checking statistical significance
    alpha = 0.05  # significance level
    is_significant = p_value < alpha

    # Display the OLS regression results using Streamlit
    st.text(results.summary())

    # Get the regression equation coefficients
    slope = round(model.coef_[0], 1)
    intercept = round(model.intercept_, 1)

    # Displaying the results in Streamlit
    st.write("-----------------------")
    st.markdown(f"<h6>A P-value should be less than 0.05 for statistical significance:</h6>", unsafe_allow_html=True)
    st.write("P-value: ", p_value)
    st.markdown(f"<h6>A higher F-statistic indicates a stronger overall significance of the model:</h6>", unsafe_allow_html=True)
    st.write("F-statistic: ", f_value)
    st.markdown(f"<h6> A good R-squared is a value > 0.9:</h6>", unsafe_allow_html=True)
    st.write("R-squared: ", r_squared)
    st.markdown(f"<h6> A Durbin-Watson Statistic value between 0 and 2 indicates no significant autocorrelation:</h6>", unsafe_allow_html=True)
    st.write("Durbin-Watson Statistic: ", durbin_watson)
    st.write("-----------------------")
    
    # Display the regression equation using Streamlit
    st.markdown(f"<h4>Weight Prediction:", unsafe_allow_html=True)
    st.markdown("")
    st.markdown(f"<h6>Predicted Weight = {slope}(day no.) + {intercept}</h6>", unsafe_allow_html=True)
    st.markdown("")
    
    # Predict weight for a specific day
    st.markdown("<h6 style='text-align: left;'>Enter a Day No. to predict weight:</h6>", unsafe_allow_html=True)
    future_day = st.number_input(" ", value=0, step=1)
    future_weight = model.predict([[future_day]])
    rounded_weight = round(future_weight[0], 1)
    st.markdown(f"<h6>Predicted weight for day {future_day}:</h6>", unsafe_allow_html=True)
    st.markdown(f"<pre>{rounded_weight}</pre>", unsafe_allow_html=True)
 
# About BSH     
elif selected_page == 'About British Short Hair':
    # Add watermark and copyright statement
    watermark = "Author: Alan Fung"
    copyright = "Copyright © 2023 Alan Fung. All rights reserved."
    
    # Display watermark and copyright statement
    st.markdown(f'<div style="position:absolute; top:10px; right:10px; font-size:12px;">{watermark}</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="position:absolute; bottom:10px; right:10px; font-size:12px;">{copyright}</div>', unsafe_allow_html=True)
    
    st.markdown("<h2 style='text-align: left;'>About British Short Hair</h2>", unsafe_allow_html=True)
    st.write("Welcome to the 'About British Shorthair' page! Here, you'll learn about the British Shorthair breed of cats.")

    st.image("https://drive.google.com/uc?export=view&id=11YXQjRzcBr29EudTUTiQO6MUBgejpp-H", width=500)
    
    st.subheader("Introduction")
    st.write("The British Shorthair is a popular breed of domestic cat known for its round face, dense coat, and sturdy build. They are medium to large-sized cats with a friendly and calm temperament.")

    st.subheader("Appearance")
    st.write("British Shorthairs have a distinct appearance characterized by their round heads, round cheeks, and round, expressive eyes. They have a short, dense coat that comes in a variety of colors and patterns.")

    st.subheader("Personality")
    st.write("British Shorthairs are known for their laid-back and gentle nature. They are generally easygoing, independent, and make great companions. They enjoy being around their human family members but are not overly demanding.")

    st.subheader("History")
    st.write("The British Shorthair is one of the oldest and most well-known cat breeds originating from the United Kingdom. They were initially bred for their hunting skills to control rodents. Over time, their popularity grew, and they became cherished pets.")

    st.subheader("Care")
    st.write("British Shorthairs are low-maintenance cats when it comes to grooming. Their dense coat requires regular brushing to prevent matting. They are generally healthy cats but may be prone to certain genetic health conditions, so regular veterinary check-ups are important.")

    st.subheader("Conclusion")
    st.write("British Shorthairs are delightful companions known for their charming looks and calm temperament. Whether you're looking for a lap cat or a furry friend to share your space with, the British Shorthair is a wonderful choice.")

# 1st Bro        
elif selected_page == '1st Bro':
    # Add watermark and copyright statement
    watermark = "Author: Alan Fung"
    copyright = "Copyright © 2023 Alan Fung. All rights reserved."
    
    # Display watermark and copyright statement
    st.markdown(f'<div style="position:absolute; top:10px; right:10px; font-size:12px;">{watermark}</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="position:absolute; bottom:10px; right:10px; font-size:12px;">{copyright}</div>', unsafe_allow_html=True)
    
    st.markdown("<h2 style='text-align: left;'>Hello Everyone, I am 1st Bro!</h2>", unsafe_allow_html=True)
    
    st.markdown(f"<h6>This is me at Day 18:</h6>", unsafe_allow_html=True)
    st.image("https://drive.google.com/uc?export=view&id=1mN1eiZRV9H4wXhSk4SbsZQGXPqjBprWU", width=250)
    
    # Set figure size and font size
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 10

    # Create a line plot
    fig, ax = plt.subplots()
    ax.plot(kwa['Day'], kwa['A (Male)'], label='A (Male)', color='#FFD700')
    ax.plot(kwa['Day'], kwa['Mean'], label='Mean', color='lightgrey', linestyle='--')
    ax.set_xlabel('Day No.')
    ax.set_ylabel('Weight (g)')
    ax.set_title('1st Bro Weight (g) by Day compare to Mean')
    ax.legend()

    # Display the chart using Streamlit
    st.pyplot(fig)
    
    # Define the predictors (X) and the target variable (y)
    Xa = kwa[['Day']]
    ya = kwa['A (Male)']

    # Create and fit the linear regression model
    model_a = LinearRegression()
    model_a.fit(Xa, ya)

    # Add constant to the predictors for statsmodels
    Xa = sm.add_constant(Xa)

    # Create and fit the OLS model
    ols_model_a = sm.OLS(ya, Xa)
    results_a = ols_model_a.fit()
    
    # Extracting statistical values
    p_value_a = results_a.pvalues['Day']
    f_value_a = round(results_a.fvalue, 1)
    r_squared_a = round(results_a.rsquared, 4)
    durbin_watson_a = round(sm.stats.stattools.durbin_watson(results_a.resid), 3)
    
    # Checking statistical significance
    alpha_a = 0.05  # significance level
    is_significant_a = p_value_a < alpha_a

    # Display the OLS regression results using Streamlit
    st.text(results_a.summary())

    # Get the regression equation coefficients
    slope_a = round(model_a.coef_[0], 1)
    intercept_a = round(model_a.intercept_, 1)

    # Displaying the results in Streamlit
    st.write("-----------------------")
    st.markdown(f"<h6>A P-value should be less than 0.05 for statistical significance:</h6>", unsafe_allow_html=True)
    st.write("P-value: ", p_value_a)
    st.markdown(f"<h6>A higher F-statistic indicates a stronger overall significance of the model:</h6>", unsafe_allow_html=True)
    st.write("F-statistic: ", f_value_a)
    st.markdown(f"<h6> A good R-squared is a value > 0.9:</h6>", unsafe_allow_html=True)
    st.write("R-squared: ", r_squared_a)
    st.markdown(f"<h6> A Durbin-Watson Statistic value between 0 and 2 indicates no significant autocorrelation:</h6>", unsafe_allow_html=True)
    st.write("Durbin-Watson Statistic: ", durbin_watson_a)
    st.write("-----------------------")

elif selected_page == '2nd Bro':
    # Add watermark and copyright statement
    watermark = "Author: Alan Fung"
    copyright = "Copyright © 2023 Alan Fung. All rights reserved."
    
    # Display watermark and copyright statement
    st.markdown(f'<div style="position:absolute; top:10px; right:10px; font-size:12px;">{watermark}</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="position:absolute; bottom:10px; right:10px; font-size:12px;">{copyright}</div>', unsafe_allow_html=True)
    
    st.markdown("<h2 style='text-align: left;'>Hello Everyone, I am 2nd Bro!</h2>", unsafe_allow_html=True)
    
    st.markdown(f"<h6>This is me at Day 18:</h6>", unsafe_allow_html=True)
    st.image("https://drive.google.com/uc?export=view&id=11baRr97Gz_V-cZTgkwSgX_5Ue_wcT11K", width=250)
    
    # Set figure size and font size
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 10

    # Create a line plot
    fig, ax = plt.subplots()
    ax.plot(kwb['Day'], kwb['B (Male)'], label='B (Male)', color='orange')
    ax.plot(kwb['Day'], kwb['Mean'], label='Mean', color='lightgrey', linestyle='--')
    ax.set_xlabel('Day No.')
    ax.set_ylabel('Weight (g)')
    ax.set_title('2nd Bro Weight (g) by Day compare to Mean')
    ax.legend()

    # Display the chart using Streamlit
    st.pyplot(fig)

        # Define the predictors (X) and the target variable (y)
    Xb = kwb[['Day']]
    yb = kwb['B (Male)']

    # Create and fit the linear regression model
    model_b = LinearRegression()
    model_b.fit(Xb, yb)

    # Add constant to the predictors for statsmodels
    Xb = sm.add_constant(Xb)

    # Create and fit the OLS model
    ols_model_b = sm.OLS(yb, Xb)
    results_b = ols_model_b.fit()
    
    # Extracting statistical values
    p_value_b = results_b.pvalues['Day']
    f_value_b = round(results_b.fvalue, 1)
    r_squared_b = round(results_b.rsquared, 4)
    durbin_watson_b = round(sm.stats.stattools.durbin_watson(results_b.resid), 3)
    
    # Checking statistical significance
    alpha_b = 0.05  # significance level
    is_significant_b = p_value_b < alpha_b

    # Display the OLS regression results using Streamlit
    st.text(results_b.summary())

    # Get the regression equation coefficients
    slope_b = round(model_b.coef_[0], 1)
    intercept_b = round(model_b.intercept_, 1)

    # Displaying the results in Streamlit
    st.write("-----------------------")
    st.markdown(f"<h6>A P-value should be less than 0.05 for statistical significance:</h6>", unsafe_allow_html=True)
    st.write("P-value: ", p_value_b)
    st.markdown(f"<h6>A higher F-statistic indicates a stronger overall significance of the model:</h6>", unsafe_allow_html=True)
    st.write("F-statistic: ", f_value_b)
    st.markdown(f"<h6> A good R-squared is a value > 0.9:</h6>", unsafe_allow_html=True)
    st.write("R-squared: ", r_squared_b)
    st.markdown(f"<h6> A Durbin-Watson Statistic value between 0 and 2 indicates no significant autocorrelation:</h6>", unsafe_allow_html=True)
    st.write("Durbin-Watson Statistic: ", durbin_watson_b)
    st.write("-----------------------")
    
elif selected_page == '3rd Sis':
    # Add watermark and copyright statement
    watermark = "Author: Alan Fung"
    copyright = "Copyright © 2023 Alan Fung. All rights reserved."
    
    # Display watermark and copyright statement
    st.markdown(f'<div style="position:absolute; top:10px; right:10px; font-size:12px;">{watermark}</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="position:absolute; bottom:10px; right:10px; font-size:12px;">{copyright}</div>', unsafe_allow_html=True)
    
    st.markdown("<h2 style='text-align: left;'>Hello Everyone, I am 3rd Sis!</h2>", unsafe_allow_html=True)
    
    st.markdown(f"<h6>This is me at Day 18:</h6>", unsafe_allow_html=True)
    st.image("https://drive.google.com/uc?export=view&id=1s2CfXVjkKuTmCdRtLIj-qJxI_Pr0Jtgd", width=250)
    
    # Set figure size and font size
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 10

    # Create a line plot
    fig, ax = plt.subplots()
    ax.plot(kwc['Day'], kwc['C (Female)'], label='C (Female)', color='red')
    ax.plot(kwc['Day'], kwc['Mean'], label='Mean', color='lightgrey', linestyle='--')
    ax.set_xlabel('Day No.')
    ax.set_ylabel('Weight (g)')
    ax.set_title('3rd Sis Weight (g) by Day compare to Mean')
    ax.legend()

    # Display the chart using Streamlit
    st.pyplot(fig)

    # Define the predictors (X) and the target variable (y)
    Xc = kwc[['Day']]
    yc = kwc['C (Female)']

    # Create and fit the linear regression model
    model_c = LinearRegression()
    model_c.fit(Xc, yc)

    # Add constant to the predictors for statsmodels
    Xc = sm.add_constant(Xc)

    # Create and fit the OLS model
    ols_model_c = sm.OLS(yc, Xc)
    results_c = ols_model_c.fit()
    
    # Extracting statistical values
    p_value_c = results_c.pvalues['Day']
    f_value_c = round(results_c.fvalue, 1)
    r_squared_c = round(results_c.rsquared, 4)
    durbin_watson_c = round(sm.stats.stattools.durbin_watson(results_c.resid), 3)
    
    # Checking statistical significance
    alpha_c = 0.05  # significance level
    is_significant_c = p_value_c < alpha_c

    # Display the OLS regression results using Streamlit
    st.text(results_c.summary())

    # Get the regression equation coefficients
    slope_c = round(model_c.coef_[0], 1)
    intercept_c = round(model_c.intercept_, 1)

    # Displaying the results in Streamlit
    st.write("-----------------------")
    st.markdown(f"<h6>A P-value should be less than 0.05 for statistical significance:</h6>", unsafe_allow_html=True)
    st.write("P-value: ", p_value_c)
    st.markdown(f"<h6>A higher F-statistic indicates a stronger overall significance of the model:</h6>", unsafe_allow_html=True)
    st.write("F-statistic: ", f_value_c)
    st.markdown(f"<h6> A good R-squared is a value > 0.9:</h6>", unsafe_allow_html=True)
    st.write("R-squared: ", r_squared_c)
    st.markdown(f"<h6> A Durbin-Watson Statistic value between 0 and 2 indicates no significant autocorrelation:</h6>", unsafe_allow_html=True)
    st.write("Durbin-Watson Statistic: ", durbin_watson_c)
    st.write("-----------------------")
    
elif selected_page == '4th Sis':
    # Add watermark and copyright statement
    watermark = "Author: Alan Fung"
    copyright = "Copyright © 2023 Alan Fung. All rights reserved."
    
    # Display watermark and copyright statement
    st.markdown(f'<div style="position:absolute; top:10px; right:10px; font-size:12px;">{watermark}</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="position:absolute; bottom:10px; right:10px; font-size:12px;">{copyright}</div>', unsafe_allow_html=True)
    
    st.markdown("<h2 style='text-align: left;'>Hello Everyone, I am 4th Sis!</h2>", unsafe_allow_html=True)
    
    st.markdown(f"<h6>This is me at Day 18:</h6>", unsafe_allow_html=True)
    st.image("https://drive.google.com/uc?export=view&id=1VI07dM7WlMiRTnVVJRIWNvtaTpy-E7sY", width=250)
    
    # Set figure size and font size
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 10

    # Create a line plot
    fig, ax = plt.subplots()
    ax.plot(kwd['Day'], kwd['D (Female)'], label='D (Female)', color='blue')
    ax.plot(kwd['Day'], kwd['Mean'], label='Mean', color='lightgrey', linestyle='--')
    ax.set_xlabel('Day No.')
    ax.set_ylabel('Weight (g)')
    ax.set_title('4th Sis Weight (g) by Day compare to Mean')
    ax.legend()

    # Display the chart using Streamlit
    st.pyplot(fig)

        # Define the predictors (X) and the target variable (y)
    Xd = kwd[['Day']]
    yd = kwd['D (Female)']

    # Create and fit the linear regression model
    model_d = LinearRegression()
    model_d.fit(Xd, yd)

    # Add constant to the predictors for statsmodels
    Xd = sm.add_constant(Xd)

    # Create and fit the OLS model
    ols_model_d = sm.OLS(yd, Xd)
    results_d = ols_model_d.fit()
    
    # Extracting statistical values
    p_value_d = results_d.pvalues['Day']
    f_value_d = round(results_d.fvalue, 1)
    r_squared_d = round(results_d.rsquared, 4)
    durbin_watson_d = round(sm.stats.stattools.durbin_watson(results_d.resid), 3)
    
    # Checking statistical significance
    alpha_d = 0.05  # significance level
    is_significant_d = p_value_d < alpha_d

    # Display the OLS regression results using Streamlit
    st.text(results_d.summary())

    # Get the regression equation coefficients
    slope_d = round(model_d.coef_[0], 1)
    intercept_d = round(model_d.intercept_, 1)

    # Displaying the results in Streamlit
    st.write("-----------------------")
    st.markdown(f"<h6>A P-value should be less than 0.05 for statistical significance:</h6>", unsafe_allow_html=True)
    st.write("P-value: ", p_value_d)
    st.markdown(f"<h6>A higher F-statistic indicates a stronger overall significance of the model:</h6>", unsafe_allow_html=True)
    st.write("F-statistic: ", f_value_d)
    st.markdown(f"<h6> A good R-squared is a value > 0.9:</h6>", unsafe_allow_html=True)
    st.write("R-squared: ", r_squared_d)
    st.markdown(f"<h6> A Durbin-Watson Statistic value between 0 and 2 indicates no significant autocorrelation:</h6>", unsafe_allow_html=True)
    st.write("Durbin-Watson Statistic: ", durbin_watson_d)
    st.write("-----------------------")
    
elif selected_page == '5th Sis':
    # Add watermark and copyright statement
    watermark = "Author: Alan Fung"
    copyright = "Copyright © 2023 Alan Fung. All rights reserved."
    
    # Display watermark and copyright statement
    st.markdown(f'<div style="position:absolute; top:10px; right:10px; font-size:12px;">{watermark}</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="position:absolute; bottom:10px; right:10px; font-size:12px;">{copyright}</div>', unsafe_allow_html=True)
    
    st.markdown("<h2 style='text-align: left;'>Hello Everyone, I am 5th Sis!</h2>", unsafe_allow_html=True)
    
    st.markdown(f"<h6>This is me at Day 18:</h6>", unsafe_allow_html=True)
    st.image("https://drive.google.com/uc?export=view&id=1ielcMv_ZkUW9vPR8Z3Kijgp05qPhh-tq", width=250)
    
    # Set figure size and font size
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 10

    # Create a line plot
    fig, ax = plt.subplots()
    ax.plot(kwe['Day'], kwe['E (Female)'], label='E (Female)', color='green')
    ax.plot(kwe['Day'], kwe['Mean'], label='Mean', color='lightgrey', linestyle='--')
    ax.set_xlabel('Day No.')
    ax.set_ylabel('Weight (g)')
    ax.set_title('5th Sis Weight (g) by Day compare to Mean')
    ax.legend()

    # Display the chart using Streamlit
    st.pyplot(fig)

    # Define the predictors (X) and the target variable (y)
    Xe = kwe[['Day']]
    ye = kwe['E (Female)']

    # Create and fit the linear regression model
    model_e = LinearRegression()
    model_e.fit(Xe, ye)

    # Add constant to the predictors for statsmodels
    Xe = sm.add_constant(Xe)

    # Create and fit the OLS model
    ols_model_e = sm.OLS(ye, Xe)
    results_e = ols_model_e.fit()
    
    # Extracting statistical values
    p_value_e = results_e.pvalues['Day']
    f_value_e = round(results_e.fvalue, 1)
    r_squared_e = round(results_e.rsquared, 4)
    durbin_watson_e = round(sm.stats.stattools.durbin_watson(results_e.resid), 3)
    
    # Checking statistical significance
    alpha_e = 0.05  # significance level
    is_significant_e = p_value_e < alpha_e

    # Display the OLS regression results using Streamlit
    st.text(results_e.summary())

    # Get the regression equation coefficients
    slope_e = round(model_e.coef_[0], 1)
    intercept_e = round(model_e.intercept_, 1)

    # Displaying the results in Streamlit
    st.write("-----------------------")
    st.markdown(f"<h6>A P-value should be less than 0.05 for statistical significance:</h6>", unsafe_allow_html=True)
    st.write("P-value: ", p_value_e)
    st.markdown(f"<h6>A higher F-statistic indicates a stronger overall significance of the model:</h6>", unsafe_allow_html=True)
    st.write("F-statistic: ", f_value_e)
    st.markdown(f"<h6> A good R-squared is a value > 0.9:</h6>", unsafe_allow_html=True)
    st.write("R-squared: ", r_squared_e)
    st.markdown(f"<h6> A Durbin-Watson Statistic value between 0 and 2 indicates no significant autocorrelation:</h6>", unsafe_allow_html=True)
    st.write("Durbin-Watson Statistic: ", durbin_watson_e)
    st.write("-----------------------")
    
elif selected_page == 'Leave Your Comments':
    # Add watermark and copyright statement
    watermark = "Author: Alan Fung"
    copyright = "Copyright © 2023 Alan Fung. All rights reserved."
    
    # Display watermark and copyright statement
    st.markdown(f'<div style="position:absolute; top:10px; right:10px; font-size:12px;">{watermark}</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="position:absolute; bottom:10px; right:10px; font-size:12px;">{copyright}</div>', unsafe_allow_html=True)
    
    st.markdown("<h2 style='text-align: center;'>Please Leave Your Comments</h2>", unsafe_allow_html=True)
    
    # Get current time
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Add a text input for users to leave their comments
    user_comment = st.text_input("Your Comment:", "")

    # Add a button to submit the comment
    if st.button("Submit Comment"):
        # Append the comment and timestamp to a file or database for record-keeping
        with open("comments.txt", "a") as file:
            file.write(f"{current_time}: {user_comment}\n")
        st.success("Your comment has been submitted.")
