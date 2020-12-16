# ARR Term Rate Calculator
# EY
# UChicago Project Lab
# for streamlit

import streamlit as st
import pandas as pd

from QuantLib import *
import pandas as pd
import numpy as np
import datetime
from scipy.optimize import minimize
import matplotlib.pyplot as plt  
from IPython.display import Image

def datetime_to_ql(d):
    d = d.split("/")
    return Date(int(d[1]), int(d[0]), int(d[2]))

def ql_to_datetime(d):
    date_dt = datetime.datetime(d.year(), d.month(), d.dayOfMonth())
    return datetime.datetime.strftime(date_dt, "%m/%d/%Y")

def getFirstDayOfMonth(d):
    return Date(1,d.month(),d.year())

def getMonthDays(d):
    return Date.endOfMonth(d).dayOfMonth()  

def get_mat_A(start_date_str, mon_len, MPC_dates_raw, futures, FFact):
    calendar = TARGET()
    start_date = datetime_to_ql(start_date_str)
    end_date = calendar.advance(start_date, Period(mon_len, Months), ModifiedFollowing)
    MPC_dates = []
    for date in MPC_dates_raw:
        if date is not None:
            MPC_dates.append(datetime_to_ql(date))

    Changing_dates = []
    Changing_dates.append(start_date)
    for mpc_date in MPC_dates:
        if mpc_date > start_date: #only append mpc date that is greater than start date    
            Changing_dates.append(mpc_date)
    Changing_dates.append(end_date)
    Changing_dates = sorted(Changing_dates)

    # column number is determined by how many changing rates
    col= len(Changing_dates) + 1

    # row number is determined by the tenor of term rate
    if end_date.year() == start_date.year():
        row = end_date.month() - start_date.month() + 2  #if is 3M term rate, 4month plus one row for FFact. Note that when different year, we need to add 12
    elif end_date.year() > start_date.year():
        row = end_date.month()+12 - start_date.month() + 2

    #Construct Matrix A:     
    #first row:   
    f0 = []
    f0.append(1)
    for idx in range(col-1):
        f0.append(0)

    #FFact = 0.345 #user input or daily realized average of historical data in that month up to start date
    f01 = (start_date - getFirstDayOfMonth(start_date)) / getMonthDays(start_date)

    # contruct future month range end
    futureRangeEnd = []
    for i in range(row-1):
        if start_date.month()+i <= 12:
            futureRangeEnd.append(Date.endOfMonth(Date(1,start_date.month()+i,start_date.year())))
        elif start_date.month()+i > 12:
            futureRangeEnd.append(Date.endOfMonth(Date(1,start_date.month()+i-12,start_date.year()+1)))
    futureRangeStart = []
    for i in range(len(futureRangeEnd)):
        futureRangeStart.append(getFirstDayOfMonth(futureRangeEnd[i]))

    f = []
    for idx in range(row-1):
        f_row = []
        last_date = Date(1,1,1901) 
        for idy, date in enumerate(Changing_dates):
            if futureRangeStart[idx] <= date <= futureRangeEnd[idx]: #if changing date within the future month
                f_row.append((date - futureRangeStart[idx])/getMonthDays(date) - sum(f_row))
                if date == Changing_dates[-1]: # if last changing point need to include the partial rate for the rest of the month
                    f_row.append(1 - sum(f_row))

            elif futureRangeStart[idx] <= last_date <= futureRangeEnd[idx]: # if changing date not fully within the future month
                f_row.append(1 - sum(f_row))
                if date == Changing_dates[-1]: # after last changing point, if date not falling current future month just append zero
                    f_row.append(0)
            else:
                f_row.append(0)
                if date == Changing_dates[-1]: # after last changing point, if date not falling current future month just append zero
                    f_row.append(0)
            last_date = date

        if sum(f_row) == 0: # check if no changing point fall within the future range
            deltas = []
            for date in Changing_dates: # get the left nearest changing date for the specific month (smallest postitive delta)
                deltas.append(futureRangeStart[idx] - date) 

            minimum = deltas[0]
            minindex = 0
            for index, delta in enumerate(deltas): # get smallest positive delta
                if delta > 0 and delta < minimum:
                    minimum = delta
                    minindex = index
            f_row[minindex + 1] = 1   #change the covering rate to 1 

        f.append(f_row)

    A = ([f0] + f)
    A = np.array(A)
    #b = np.array([0.4531, 0.455, 0.465, 0.605, 0.605]) # 3M
    #b = np.array([0.4531, 0.455, 0.465]) #1M
    # FFact: historic avg
    
    tmp = np.array([FFact])
    b = np.concatenate((tmp, futures), axis = 0)
    
    i = 0
    while i > -A.shape[1]:
        if np.sum(np.abs(A[:, i-1])) == 0:
            i -= 1
        else:
            break
    if i == 0:
        return A, b[:A.shape[0]]
    else:
        return A[:, :i], b[:A.shape[0]]


def fun(x, A, b, M): 
    b = b[:A.shape[0]]
    epsilon = np.linalg.norm(np.dot(A,x) - b)
    penalty = 0
    for idx in range(0,len(x)-2):
        penalty = penalty + (x[idx+2] - 2*x[idx+1] + x[idx])**2
    #print('penalty is')    
    #print(penalty)
    #print('epsilon is')
    #print(epsilon)
    return ((1-M)*penalty + M*epsilon)

def solve_eq(A, b, M):
    n = A.shape[1]
    sol = minimize(fun, np.zeros(n), args = (A,b,M), method='L-BFGS-B', bounds=[(0.,None) for x in np.arange(n)])
    x = sol['x']
    return x

def calculate_term_rate(start_date, mon_len, MPC_dates_raw, x):

    calendar = TARGET()
    start_date = datetime_to_ql(start_date)
    end_date = calendar.advance(start_date, Period(mon_len, Months), ModifiedFollowing)

    term_dates = [] #create datafrome for all dates used for claculate termrate
    annual_rates = [] # SOFR rate from solution x 
    daily_rates = []
    daycount_fraction_denominator = 365
    target_days = end_date - start_date

    MPC_dates = []
    for date in MPC_dates_raw:
        if date is not None:
            MPC_dates.append(datetime_to_ql(date))

    Changing_dates = []
    Changing_dates.append(start_date)
    for mpc_date in MPC_dates:
        if mpc_date > start_date and mpc_date < end_date: #only append mpc date that is greater than start date    
            Changing_dates.append(mpc_date)
    Changing_dates.append(end_date)
    Changing_dates = sorted(Changing_dates)

    df_end = Date.endOfMonth(end_date)
    df_start = getFirstDayOfMonth(start_date)
    calendar_days = df_end - df_start

    for idx in range (calendar_days+1):
        term_dates.append(df_start + idx)
        daily_rates.append(np.nan)
        annual_rates.append(np.nan)

    d = {'date': term_dates, 'annual_rate': annual_rates, 'daily_rate': daily_rates}    
    df = pd.DataFrame(data = d)
    #df = df.set_index('date')
    frames = []
    temp = df[df.date < Changing_dates[0]]
    temp.annual_rate = x[0]
    frames.append(temp)
    for idx in range(len(Changing_dates)-1):
        temp = df[df.date >= Changing_dates[idx]][df.date < Changing_dates[idx+1]]
        temp.annual_rate = x[idx+1]
        #df[(df.date >= Changing_dates[idx]) & (df.date < Changing_dates[idx+1])]['annual_rate'] = x[idx+1] #why this does not work? need deep copy?
        frames.append(temp)
    result = pd.concat(frames)
    
    for idx, row in result.iterrows(): #need to add idx here to make row not a tuple
        if row.date < end_date and row.date>= start_date:
            row['daily_rate'] = 1 + row['annual_rate']/100*1/daycount_fraction_denominator
            #print(row.daily_rate)
            result.at[idx, 'daily_rate'] = row['daily_rate'] #update the dataframe using .at (previously .set_value)

    final = result.dropna()
    term_rate = (final.daily_rate.prod() - 1) * daycount_fraction_denominator/target_days * 100
    return term_rate


###############################################################

cme_ans = np.array([[0.0656, 0.0602, 0.0560],#12-15
 [0.0638, 0.0579, 0.0554],#14
 [0.0675, 0.0620, 0.0594],#11
 [0.0645, 0.0601, 0.0583],#10
 [0.0671, 0.0633, 0.0598]])#9

ice_ans = np.array([[0.0691, 0.0615, 0.0561],#15
 [0.0648, 0.0566, 0.0533],#14
 [0.0664, 0.0653, 0.0619],#11
 [0.0669, 0.0638, 0.0612],#10
 [0.0689, 0.0640, 0.0605]])#9

MPC_dates_raw = ["11/05/2020", "12/16/2020", "01/27/2021", "03/17/2021", "04/28/2021"]
# https://www.cmegroup.com/trading/interest-rates/stir/one-month-sofr_quotes_settlements_futures.html#tradeDate=11%2F20%2F2020
futures_all = [[99.9250, 99.9350, 99.9400, 99.9450, 99.9450, 99.9500, 99.9450],
 [99.9250, 99.9350, 99.9400, 99.9450, 99.9450, 99.9500, 99.9450],
 [99.9275, 99.9400, 99.9450, 99.9450, 99.9450, 99.9500, 99.9450],
 [99.9275, 99.9350, 99.9400, 99.9400, 99.9400, 99.9450, 99.9400],
 [99.9275, 99.9350, 99.9400, 99.9400, 99.9400, 99.9450, 99.9400]]
# https://apps.newyorkfed.org/markets/autorates/SOFR#Chart12
sofr_raw = [0.08, 0.08, 0.08, 0.09, 0.08, \
 0.07, 0.08, 0.08, 0.08, 0.08]


date_l = ["12/9/2020", "12/10/2020", "12/11/2020", "12/14/2020","12/15/2020"]

terms = ['1M','3M','6M']

def calc_term_rate_list(M):
	array = []
	for month_len in [1, 3, 6]:
	    tmp_l = []
	    for i in range(len(date_l)):
	        FFact = np.mean(sofr_raw[:16+i])
	        futures_raw = np.array(futures_all[i])
	        futures = []
	        for quote in futures_raw:
	            if quote is not None:
	                futures.append(100-quote)
	        A, b = get_mat_A(date_l[i], month_len, MPC_dates_raw, futures, FFact)
	        x = solve_eq(A, b, M)
	        tmp_l.append(calculate_term_rate(date_l[i], month_len, MPC_dates_raw, x))
	    array.append(tmp_l)
	return array

###############################################################
st.write("""
# EY SOFR Calculate App Demo

This app present the term rate of SOFR based on the future implied
term rate estimation approach proposed by ARRC.
""")

st.sidebar.header('User Input Parameters')

def user_input_features():

	M = st.sidebar.slider('Optimization Weight',0.0,1.0,0.6)
	date = st.sidebar.selectbox("Evaluation Date", date_l)
	mon_len = st.sidebar.selectbox('Term',terms)

	data = {'Optimization Weight': M,
			'Evaluation Date': date,
			'Term': mon_len}
	features = pd.DataFrame(data,index=[0])
	return features


df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

M = df.values[0,0]
date = df.values[0,1]
term = df.values[0,2]
array = calc_term_rate_list(M)

st.subheader('Calculate term rate')
st.write(array[terms.index(term)][date_l.index(date)])

st.subheader('Report')
report_data = {'Rate Changing Dates': MPC_dates_raw,
			}
st.write(pd.DataFrame(report_data))

st.subheader('Comparison with CME/ICE - '+df.values[0,2]+' term rate')
chart_data = pd.DataFrame({
 'date': date_l,
 'CME': cme_ans[:len(date_l)].T[terms.index(term)],
 'ICE': ice_ans[:len(date_l)].T[terms.index(term)],
 'ours': array[terms.index(term)]
 })
chart_data.set_index('date',inplace=True)

st.line_chart(chart_data)
