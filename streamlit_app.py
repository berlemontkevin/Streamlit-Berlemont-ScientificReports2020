import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px


st.title("Nonlinear neural network dynamics accounts for human confidence in a sequence of perceptual decisions - Berlemont et al., Scientific Reports 2020")

st.markdown(''' Electrophysiological recordings during perceptual decision tasks in monkeys suggest that the degree of confidence in a decision is based on a simple neural signal produced by the neural decision process. Attractor neural networks provide an appropriate biophysical modeling framework, and account for the experimental results very well. However, it remains unclear whether attractor neural networks can account for confidence reports in humans. We present the results from an experiment in which participants are asked to perform an orientation discrimination task, followed by a confidence judgment. Here we show that an attractor neural network model quantitatively reproduces, for each participant, the relations between accuracy, response times and confidence. We show that the attractor neural network also accounts for confidence-specific sequential effects observed in the experiment (participants are faster on trials following high confidence trials). Remarkably, this is obtained as an inevitable outcome of the network dynamics, without any feedback specific to the previous decision (that would result in, e.g., a change in the model parameters before the onset of the next trial). Our results thus suggest that a metacognitive process such as confidence in one’s decision is linked to the intrinsically nonlinear dynamics of the decision-making neural network.
''')


df = pd.read_csv('data/Manip3.csv')

fig = px.box(df, y ='Rt1', x= 'Name', color = 'Name')
st.plotly_chart(fig)


st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
name = st.radio("Participant",[1,2,3,5,6,7,8])

col1,col2 = st.columns(2)
# perform groupby
res = df[df['Name']== name]
fig2 = px.histogram(res, x = 'Resp2', color = 'Name')
# fig2.update_traces(marker = dict(size=14))
# res_temp = res.groupby().mean.rese
res['absAngle'] = abs(res['AngleValue'])
fig2.update_layout(width=500,
    height=500)
res_temp = res.groupby('absAngle')[['Rt1', 'Acc']].mean().reset_index()
res_temp['Acc'] += 1
res_temp['Acc'] /=2
fig3 = px.line(res_temp, x = 'absAngle', y = 'Acc', markers = True)
fig3.update_layout(width=400,
    height=500)
# fig2.update_traces(marker = dict(size=14))
col1.plotly_chart(fig3)
col2.plotly_chart(fig2)


total_df = pd.DataFrame(columns = ['Participant', 'Confidence level', 'RT(s)', 'Acc'])

for i in range(6):
    j = i+1
    temp = pd.read_csv('data/RT'+str(j)+'.csv')
    tempA = pd.read_csv('data/Acc'+str(j)+'.csv')

    for k in range(10):
        t = 'C' + str(k)
        temp_df = pd.DataFrame(data = {'Participant':[j], 'Confidence level':[k], 'RT(s)':[temp[t].mean()/1000], 'Acc':[(tempA[t].mean()+1)/2]})
        total_df= pd.concat([total_df,temp_df]) 


fig_temp = px.line(total_df, x = 'Confidence level', y='RT(s)', color='Participant',facet_col='Participant', facet_col_wrap = 3, markers = True)
st.plotly_chart(fig_temp)


fig_temp_acc = px.line(total_df, x = 'Confidence level', y='Acc', color='Participant',facet_col='Participant', facet_col_wrap = 3, markers = True)
st.plotly_chart(fig_temp_acc)










