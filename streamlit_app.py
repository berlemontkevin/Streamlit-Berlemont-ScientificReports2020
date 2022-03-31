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

st.markdown(''' Electrophysiological recordings during perceptual decision tasks in monkeys suggest that the degree of confidence in a decision is based on a simple neural signal produced by the neural decision process. Attractor neural networks provide an appropriate biophysical modeling framework, and account for the experimental results very well. However, it remains unclear whether attractor neural networks can account for confidence reports in humans. We present the results from an experiment in which participants are asked to perform an orientation discrimination task, followed by a confidence judgment. Here we show that an attractor neural network model quantitatively reproduces, for each participant, the relations between accuracy, response times and confidence. We show that the attractor neural network also accounts for confidence-specific sequential effects observed in the experiment (participants are faster on trials following high confidence trials). Remarkably, this is obtained as an inevitable outcome of the network dynamics, without any feedback specific to the previous decision (that would result in, e.g., a change in the model parameters before the onset of the next trial). Our results thus suggest that a metacognitive process such as confidence in one’s decision is linked to the intrinsically nonlinear dynamics of the decision-making neural network.
''')



# Descirption of the data
df = pd.read_csv('data/Manip3.csv')

fig = px.box(df, y ='Rt1', x= 'Name', color = 'Name')
st.plotly_chart(fig)


st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
name = st.radio("Participant Number",[1,2,3,5,6,7,8])

# Plot RTs and ACC of the Participants
col1,col2 = st.columns(2)
# perform groupby
res = df[df['Name']== name]
fig2 = px.histogram(res, x = 'Resp2', color = 'Name')
res['absAngle'] = abs(res['AngleValue'])
fig2.update_layout(width=500, height=500)
res_temp = res.groupby('absAngle')[['Rt1', 'Acc']].mean().reset_index()
res_temp['Acc'] += 1
res_temp['Acc'] /=2
fig3 = px.line(res_temp, x = 'absAngle', y = 'Acc', markers = True)
fig3.update_layout(width=400,
    height=500)
col1.plotly_chart(fig3)
col2.plotly_chart(fig2)
####################################

# Model example

coh = st.slider('Coh', -30, 30, step = 1)

modelparams['gE'] = st.sidebar.slider('gE', 0., 0.4, step = 0.01, value = 0.2609)
modelparams['gI'] = st.sidebar.slider('gI', 0.1, 0.0, step = 0.01, value = -0.0497)
modelparams['mu0'] = st.sidebar.slider('mu0', 0., 40., step = 1., value = 20.0)

st.sidebar.markdown('---')
st.sidebar.markdown('## Time Parameters')
modelparams['Ttotal'] = st.sidebar.slider('Ttotal', 0., 2., step = 0.1, value = 2.)
modelparams['Tstim_on'] = st.sidebar.slider('Tstim_on', 0., 1., step = 0.1, value = 0.1)
modelparams['Tstim_off'] = st.sidebar.slider('Tstim_on', 1., 2., step = 0.1, value = 1.)


model = Model(modelparams)
model.run(coh=coh, n_trial=1)
temp_df_model = pd.DataFrame(data = {'t':model.t, 'r1':model.r1, 'r2':model.r2, 'I1':model.I1, 'I2':model.I2})


fig_model = make_subplots(rows=1, cols=1)

fig_model.add_trace(
    go.Scatter(x=model.t, y=model.r1, mode = 'lines', name = 'r1'),
    row=1, col=1)
fig_model.add_trace(
    go.Scatter(x=model.t, y=model.r2, mode = 'lines', name = 'r2'),
    row=1, col=1)

# fig_model.add_trace(
#     go.Scatter(x=model.r1, y=model.r2, mode = 'lines'),
#     row=1, col=2)
# fig_model.update_xaxes(title_text="xaxis 2 title", range=[0, 35], row=1, col=2)
# fig_model.update_yaxes(title_text="xaxis 2 title", range=[0, 35], row=1, col=2)

st.plotly_chart(fig_model)
   

############

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










