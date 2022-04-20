import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
# Creation of the WOng Wang model


def F(I, a=270., b=108., d=0.154):
    """F(I) for vector I"""
    return (a*I - b)/(1.-np.exp(-d*(a*I - b)))

st.sidebar.subheader("Parameters of the model")


modelparams = dict(
    gE          = 0.2609,
    gI          = -0.0497, # cross-inhibition strength [nA]
    I0          = 0.3255, # background current [nA]
    tauS        = 0.1, # Synaptic time constant [sec]
    gamma       = 0.641, # Saturation factor for gating variable
    tau0        = 0.002, # Noise time constant [sec]
    sigma       = 0.02, # Noise magnitude [nA]
    mu0         = 20., # Stimulus firing rate [Hz]
    Jext        = 0.52, # Stimulus input strength [pA/Hz]
    Ttotal      = 2., # Total duration of simulation [s]
    Tstim_on    = 0.1, # Time of stimulus onset
    Tstim_off   = 1.0, # Time of stimulus offset
    dt          = 0.5/1000, # Simulation time step
    record_dt   = 2/1000.,
    thr = 20.0 # threshold
)

class Model(object):
    def __init__(self, modelparams):
        # Model parameters
        self.params = modelparams.copy()

    def run(self, n_trial=1, coh=0):
        p = self.params

        # Set random seed
        #np.random.seed(10)

        # Number of time points
        NT = int(p['Ttotal']/p['dt'])
        t_plot = np.arange(NT)*p['dt']
        t_stim = (t_plot>p['Tstim_on']) * (t_plot<p['Tstim_off'])

        mean_stim = np.ones(NT)*p['mu0']*p['Jext']/1000 # [nA]
        diff_stim = p['Jext']*p['mu0']*coh/100.*2
        Istim1_plot = (mean_stim + diff_stim/2/1000) * t_stim # [nA]
        Istim2_plot = (mean_stim - diff_stim/2/1000) * t_stim

        # Initialize S1 and S2
        S1 = 0.1*np.ones(n_trial)
        S2 = 0.1*np.ones(n_trial)

        Ieta1 = np.zeros(n_trial)
        Ieta2 = np.zeros(n_trial)

        n_record = int(p['record_dt']//p['dt'])
        i_record = 0
        N_record = int(p['Ttotal']/p['record_dt'])
        self.r1 = np.zeros(N_record)
        self.r2 = np.zeros(N_record)
        self.t  = np.zeros(N_record)
        self.I1 = np.zeros(N_record)
        self.I2 = np.zeros(N_record)

        # Loop over time points in a trial
        for i_t in range(NT):
            # Random dot stimulus
            Istim1 = Istim1_plot[i_t]
            Istim2 = Istim2_plot[i_t]

            # Total synaptic input
            Isyn1 = p['gE']*S1 + p['gI']*S2 + Istim1 + Ieta1
            Isyn2 = p['gE']*S2 + p['gI']*S1 + Istim2 + Ieta2

            # Transfer function to get firing rate

            r1  = F(Isyn1)
            r2  = F(Isyn2)

            #---- Dynamical equations -------------------------------------------

            # Mean NMDA-mediated synaptic dynamics updating
            S1_next = S1 + p['dt']*(-S1/p['tauS'] + (1-S1)*p['gamma']*r1)
            S2_next = S2 + p['dt']*(-S2/p['tauS'] + (1-S2)*p['gamma']*r2)

            # Ornstein-Uhlenbeck generation of noise in pop1 and 2
            Ieta1_next = Ieta1 + (p['dt']/p['tau0'])*(p['I0']-Ieta1) + np.sqrt(p['dt']/p['tau0'])*p['sigma']*np.random.randn(n_trial)
            Ieta2_next = Ieta2 + (p['dt']/p['tau0'])*(p['I0']-Ieta2) + np.sqrt(p['dt']/p['tau0'])*p['sigma']*np.random.randn(n_trial)

            S1 = S1_next
            S2 = S2_next
            Ieta1 = Ieta1_next
            Ieta2 = Ieta2_next

            if np.mod(i_t,n_record) == 1:
                self.r1[i_record] = r1
                self.r2[i_record] = r2
                self.I1[i_record] = Istim1
                self.I2[i_record] = Istim2
                self.t[i_record] = i_t*p['dt']
                i_record += 1
            # if r1 > p['thr'] or r2 > p['thr']:
            #     break

######################





st.title("Nonlinear neural network dynamics accounts for human confidence in a sequence of perceptual decisions - Berlemont et al., Scientific Reports 2020")


st.markdown(''' I designed this Streamlit app to provide an interactive visualization of the data of **Nonlinear neural network dynamics accounts for human confidence in a sequence of perceptual decisions - Berlemont et al., Scientific Reports 2020** and the dynamics of the associated model. More details on the experimental setup and the model can be found:
            
            - in the paper online: https://www.nature.com/articles/s41598-020-63582-8
            
            - and the code on github: https://github.com/berlemontkevin/Confidence_NeuralNetwork_ScientificReports2020
            
The presentation will be divided in two parts. First, I will show an interactive model of the model. Then, I will describe and show the experimental results.
            
            
Electrophysiological recordings during perceptual decision tasks in monkeys suggest that the degree of confidence in a decision is based on a simple neural signal produced by the neural decision process. Attractor neural networks provide an appropriate biophysical modeling framework, and account for the experimental results very well. However, it remains unclear whether attractor neural networks can account for confidence reports in humans. We present the results from an experiment in which participants are asked to perform an orientation discrimination task, followed by a confidence judgment.
            
            ''')




# Model example

st.subheader(''' Brief description of the model''')

st.markdown(''' The neural network model I consider is composed of two neural pools (figure below) that are selective to the two different choices. In other words, during the decision task, both neural populations receive evidence selective to one of the two options. The dynamics is a winner-take-all dynamics meaning that  after some time one of the population is active and the other isn't. The one active corresponds to the decision made by the network in the decision task. 
            
            
The amount of evidence in favor of decision 1 or decision 2 is represented by the coherence level *c*.''')

col1, col2, col3 = st.columns([2,4,2])
with col1:
    st.write("")

with col2:
    st.image('./fig/fig_attractor.png', caption = 'Schematic of the attractor neural network')
with col3:
    st.write("")

st.markdown('''Thus, population *i* receives the following stimulus: ''')
st.latex(r''' I_{stim,i}=  J_{ext} \left( 1 \pm c_{\theta} \right)
''')

st.markdown(''' The left sidebar, and the one belows modify the parameters of the neural network. The red curve corresponds to decision 2 (coherence level negative) and the blue one to a positive coherence level (decision 1). By sliding the coherence level towards the right, one can see that  the model switches its decision towards decision 1, as the network receives more evidence for this choice.''')

coh = st.slider('Coherence level', -30, 30, step = 1)

modelparams['gE'] = st.sidebar.slider('gE (nA)', 0., 0.4, step = 0.01, value = 0.2609)
modelparams['gI'] = st.sidebar.slider('gI (nA)', 0.1, 0.0, step = 0.01, value = -0.0497)
modelparams['mu0'] = st.sidebar.slider('mu0 (Hz)', 0., 40., step = 1., value = 20.0)

st.sidebar.markdown('---')
st.sidebar.markdown('## Time Parameters')
modelparams['Ttotal'] = st.sidebar.slider('Ttotal (seconds)', 0., 2., step = 0.1, value = 2.)
modelparams['Tstim_on'] = st.sidebar.slider('Tstim_on (seconds)', 0., 1., step = 0.1, value = 0.1)
modelparams['Tstim_off'] = st.sidebar.slider('Tstim_on (seconds)', 1., 2., step = 0.1, value = 1.)


model = Model(modelparams)
model.run(coh=coh, n_trial=1)
temp_df_model = pd.DataFrame(data = {'t':model.t, 'r1':model.r1, 'r2':model.r2, 'I1':model.I1, 'I2':model.I2})


fig_model = make_subplots(rows=1, cols=1)

fig_model.add_trace(
    go.Scatter(x=model.t, y=model.r1, mode = 'lines', name = 'Population 1'),
    row=1, col=1)
fig_model.add_trace(
    go.Scatter(x=model.t, y=model.r2, mode = 'lines', name = 'Population 2'),
    row=1, col=1)
fig_model.update_layout(
    title="Neural activity timecourse",
    xaxis_title="Time (s)",
    yaxis_title="Firing Rate (Hz)")

st.plotly_chart(fig_model)
   

############


st.subheader(''' Experimental results''')

st.markdown(''' In the experiments, particpants have to look at a screen and report if the orientation of the Gabor patch is clockwise or counterclockwise. Moreover, at the end of the trials, participants report their confidence on a scale from 1 to 10. ''')

st.image('./fig/exp-procedure.png', caption = 'Description of the experimental setup. More details in the published version of the paper.')

st.markdown('''The data cleaning procedure is detailed in the Methods section of the *Scientific Reports* research article. The following figure represents the reaction time of all the participants (across all trial conditions).  ''')
# Descirption of the data
df = pd.read_csv('data/Manip3.csv')

fig = px.box(df, y ='Rt1', x= 'Name', color = 'Name',
             labels = {'Rt1' : 'Reaction time (s)',
                       'Name' : 'Participant Number'},
             title = 'Reaction time of the participants')
st.plotly_chart(fig)




st.markdown('''Individual performances and confidence distribution of the participants can be selected on the figure below. ''')

st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
name = st.radio("Participant Number",[1,2,3,5,6,7,8,9])

# Plot RTs and ACC of the Participants
col1,col2 = st.columns(2)
# perform groupby
res = df[df['Name']== name]
fig2 = px.histogram(res, x = 'Resp2', color = 'Name',
                    labels = {'Resp2' : 'Confidence level',
                              'Name' : 'Participant Number'},
                    title = 'Confidence distribution')
res['absAngle'] = abs(res['AngleValue'])
fig2.update_layout(width=500, height=500)
res_temp = res.groupby('absAngle')[['Rt1', 'Acc']].mean().reset_index()
res_temp['Acc'] += 1
res_temp['Acc'] /=2
fig3 = px.line(res_temp, x = 'absAngle', y = 'Acc', markers = True,
               labels = {'absAngle' : 'Absolute value of orientation (degree)',
                         'Acc' : 'Accuracy'},
               title = 'Accuracy vs. orientation')
fig3.update_layout(width=400,
    height=500)
col1.plotly_chart(fig3)
col2.plotly_chart(fig2)
####################################


st.subheader('Model and Experiment comparison')



st.markdown('''One of the main research result of my article is the relation between accuracy, reaction time and confidence from a modeling point of view. One can observe that the model is matching the experimental results for all the participants. The mainn results being the following:
            
            - Reaction time decreases with confidence level
            
            - Accuracy increases with confidence level
            
            
If you would like more information on the model,data or more detailed figures, don't hesitate to check the academic publication as this streamlit app is to give an overview of the model, data and results only.
            ''')



# st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
j = st.radio("Participant Number",[1,3,5,6,7,9], key = 1)


total_df = pd.DataFrame(columns = ['Participant', 'Confidence level', 'RT(s)', 'Acc', 'Exp:'])


df = pd.read_csv('data/Manip3.csv')

df_n = df[df['Name'] == j]



temp = pd.read_csv('data/RT'+str(j)+'.csv')
tempA = pd.read_csv('data/Acc'+str(j)+'.csv')

for k in range(10):
        t = 'C' + str(k)
        temp_df_n = df_n[df_n['Resp2'] == k]
        temp_df = pd.DataFrame(data = {'Participant':[j], 'Confidence level':[k], 'RT(s)':[temp[t].mean()/1000], 'Acc':[(tempA[t].mean()+1)/2], 'Exp:' : 'Model'})
        total_df= pd.concat([total_df,temp_df])
        temp_df = pd.DataFrame(data = {'Participant':[j], 'Confidence level':[k], 'RT(s)':[temp_df_n['Rt1'].mean()], 'Acc':[(temp_df_n['Acc'].mean()+1)/2], 'Exp:' : 'Data'})
        total_df= pd.concat([total_df,temp_df])


fig_temp = go.Figure()
temp_total = total_df[total_df['Exp:'] == 'Model']
fig_temp.add_trace(go.Scatter(x = temp_total['Confidence level'], y = temp_total['RT(s)'], mode = 'lines', name = 'Model'))
temp_total = total_df[total_df['Exp:'] == 'Data']
fig_temp.add_trace(go.Scatter(x = temp_total['Confidence level'], y = temp_total['RT(s)'], mode = 'markers', name = 'Experimental data'))
fig_temp.update_layout(
    title="Participant " +str(j) + ": Reaction time vs. Confidence level",
    xaxis_title="Confidence level",
    yaxis_title="Reaction Time (s)")
# fig_temp = px.line(total_df, x = 'Confidence level', y='RT(s)', color='Participant',facet_col='Participant', facet_col_wrap = 3, markers = True, style = 'Exp:')
st.plotly_chart(fig_temp)


fig_temp_acc = go.Figure()
temp_total = total_df[total_df['Exp:'] == 'Model']
fig_temp_acc.add_trace(go.Scatter(x = temp_total['Confidence level'], y = temp_total['Acc'], mode = 'lines', name = 'Model'))
temp_total = total_df[total_df['Exp:'] == 'Data']
fig_temp_acc.add_trace(go.Scatter(x = temp_total['Confidence level'], y = temp_total['Acc'], mode = 'markers', name = 'Experimental data'))
fig_temp_acc.update_layout(
    title="Participant " +str(j) + ": Accuracy vs. Confidence level",
    xaxis_title="Confidence level",
    yaxis_title="Accuracy")
st.plotly_chart(fig_temp_acc) 










