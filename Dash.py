#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px


# In[3]:


df=pd.read_csv("heart.csv")


# In[4]:


df


# In[5]:


df.head(3)


# In[6]:


df.columns


# In[7]:


#data cleaning
df.dtypes


# In[8]:


#changing the data type of categorical variables
df['sex'] = df.sex.astype(object)
df['cp'] = df.cp.astype(object)
df['fbs'] = df.fbs.astype(object)
df['exang'] = df.exang.astype(object)
df['restecg'] = df.restecg.astype(object)
df['slope'] = df.slope.astype(object)
df['thal'] = df.thal.astype(object)
df['target'] = df.target.astype(object)


# In[9]:


df.dtypes


# In[10]:


#checking for data missrepesentation
df.nunique()


# In[11]:


#ca can only contain 4 variables
df['ca'].unique()


# In[12]:


df.loc[df['ca']==4,'ca']=np.NaN


# In[13]:


df['ca'].unique()


# In[14]:


#ca can only contain 3 variables
df['thal'].unique()


# In[15]:


df.loc[df['thal']==0,'thal']=np.NaN
df['thal'].unique()


# In[16]:


df.isnull().sum()


# In[17]:


#cheakig missing values
df.isnull().sum()


# In[18]:


#cheakig missing values and replace with the medi
df = df.fillna(df.median())
df.isnull().sum()


# In[19]:


#define any outliers
continous_features = ['age','trestbps','chol','thalach','oldpeak']  
def outliers(df_out, drop = False):
    for each_feature in df_out.columns:
        feature_data = df_out[each_feature]
        Q1 = np.percentile(feature_data, 25.) # 25th percentile of the data of the given feature
        Q3 = np.percentile(feature_data, 75.) # 75th percentile of the data of the given feature
        IQR = Q3-Q1 #Interquartile Range
        outlier_step = IQR * 1.5 #That's we were talking about above
        outliers = feature_data[~((feature_data >= Q1 - outlier_step) & (feature_data <= Q3 + outlier_step))].index.tolist()  
        if not drop:
            print('For the feature {}, No of Outliers is {}'.format(each_feature, len(outliers)))
        if drop:
            df.drop(outliers, inplace = True, errors = 'ignore')
            print('Outliers from {} feature removed'.format(each_feature))

outliers(df[continous_features])


# In[20]:


#drop outliers
outliers(df[continous_features],drop=True)


# In[21]:


df['target'] = df.target.replace({1: "Disease", 0: "No_disease"})
df['sex'] = df.sex.replace({1: "Male", 0: "Female"})
df['exang'] = df.exang.replace({1: "Yes", 0: "No"})
df['fbs'] = df.fbs.replace({1: "True", 0: "False"})
df['slope'] = df.slope.replace({0: "upsloping", 1: "flat",2:"downsloping"})
df['thal'] = df.thal.replace({2: "fixed_defect", 3: "reversable_defect", 1:"normal"})
df


# In[22]:


#statistical summary
df.describe()


# In[23]:


df1=pd.pivot_table(df, index=['sex'], columns=['target'],  values=['age'],aggfunc='count')
df1


# In[24]:


import plotly.graph_objects as go

Gender = ["Female", "Male"]

fig2 = go.Figure()
fig2.add_trace(go.Bar(
    x=Gender,
    y=[214,294],
    name='Disease',
    marker_color='indianred'
))
fig2.add_trace(go.Bar(
    x=Gender,
    y=[61,399],
    name='No_disease',
    marker_color='lightsalmon'
))

# Here we modify the tickangle of the xaxis, resulting in rotated labels.
fig2.update_layout(
    title='Gender variation with the Target',
    xaxis=dict(
        title='Gender',
        titlefont_size=16,
        tickfont_size=14,
    ),
    yaxis=dict(
        title='Count',
        titlefont_size=16,
        tickfont_size=14,
    ),
    
    barmode='group')
fig2.update_layout(
    {"plot_bgcolor":'#111111', "paper_bgcolor":'#111111','font': {'color': '#7FDBFF'}
    } 
)
fig2.show()


# In[25]:


df5=pd.pivot_table(df, index=['sex'], columns=['thal'],  values=['age'],aggfunc='count')
df5


# In[26]:


thal = ['fixed_defect','normal','reversable_defect']

fig1 = go.Figure()
fig1.add_trace(go.Bar(
    x=thal,
    y=[239,4,32],
    name='Female',
    marker_color='#F35A9B'
))
fig1.add_trace(go.Bar(
    x=thal,
    y=[292,60,341],
    name='Male',
    marker_color='#5A6FF3'
))

fig1.update_layout(
    title='Gender variation with the Thal',
    xaxis=dict(
        title='thal',
        titlefont_size=16,
        tickfont_size=14,
    ),
    yaxis=dict(
        title='Count',
        titlefont_size=16,
        tickfont_size=14,
    ),
   
    barmode='group')
fig1.update_layout(
    {"plot_bgcolor":'#111111', "paper_bgcolor":'#111111','font': {'color': '#7FDBFF'}
    } 
)
fig1.show()


# In[27]:


df6=pd.pivot_table(df, index=['fbs'], columns=['target'],  values=['age'],aggfunc='count')
df6


# In[28]:


fbs = ['True','False']

fig4 = go.Figure()
fig4.add_trace(go.Bar(
    x=fbs,
    y=[71,389],
    name='No_disease',
    marker_color='lightsalmon'
))
fig4.add_trace(go.Bar(
    x=fbs,
    y=[389,443],
    name='Disease',
    marker_color='indianred'
))

fig4.update_layout(
    title='FBS variation with the target',
    
    xaxis=dict(
        title='FBS',
        titlefont_size=16,
        tickfont_size=14,
    ),
    yaxis=dict(
        title='Count',
        titlefont_size=16,
        tickfont_size=14,
    ),    
    barmode='group')
fig4.update_layout(
    {"plot_bgcolor":'#111111', "paper_bgcolor":'#111111','font': {'color': '#7FDBFF'}
    } 
)
fig4.show()


# In[29]:


df2=df[df['target']=='Disease']
df2


# In[30]:


fig5=go.Figure()
fig5.add_trace(go.Histogram(
    x=df2['age'],
     name='Age',
     marker_color='#EE82EE'
))
fig5.update_layout(
    title='    Age Distribution of Patients',
    
    xaxis=dict(
        title='Age',
        titlefont_size=16,
        tickfont_size=14,
    ),
    yaxis=dict(
        title='Count',
        titlefont_size=16,
        tickfont_size=14,
    ),
)             
    
fig5.update_layout(
    {"plot_bgcolor":'#111111', "paper_bgcolor":'#111111','font': {'color': '#7FDBFF'}
    } 
)
fig5.show()


# In[31]:


df3=pd.pivot_table(df2, index=['sex'], columns=['cp'],  values=['age'],aggfunc='count')
df3


# In[32]:


cp = ["0", "1", "2", "3"]

fig3 = go.Figure()
fig3.add_trace(go.Bar(
    x=cp,
    y=[52, 50, 99, 13],
    name='Female',
    marker_color='#F35A9B'
))
fig3.add_trace(go.Bar(
    x=cp,
    y=[ 64, 84, 111 ,35],
    name='Male',
    marker_color='#5A6FF3'
))


fig3.update_layout(
    title='Chest pain type variation with the Gender',
    
    xaxis=dict(
        title='chest pain type',
        titlefont_size=16,
        tickfont_size=14,
    ),
    yaxis=dict(
        title='Count',
        titlefont_size=16,
        tickfont_size=14,
    ),
          
    barmode='group')
fig3.update_layout(
    {"plot_bgcolor":'#111111', "paper_bgcolor":'#111111','font': {'color': '#7FDBFF'}
    } 
)
fig3.show()


# In[33]:


df8=pd.pivot_table(df, index=['exang'], columns=['target'],  values=['age'],aggfunc='count')
df8


# In[34]:


target = ["Disease", "No-Disease"]

fig8 = go.Figure()
fig8.add_trace(go.Bar(
    x=target,
    y=[440,211],
    name='No',
    marker_color='#F35A9B'
))
fig8.add_trace(go.Bar(
    x=target,
    y=[ 68,249],
    name='Yes',
    marker_color='#5A6FF3'
))


fig8.update_layout(
    title='Impact of doing excercise with having heart disease ',
    
    xaxis=dict(
        title='Result',
        titlefont_size=16,
        tickfont_size=14,
    ),
    yaxis=dict(
        title='Count',
        titlefont_size=16,
        tickfont_size=14,
    ),
          
    barmode='group')
fig8.update_layout(
    {"plot_bgcolor":'#111111', "paper_bgcolor":'#111111','font': {'color': '#7FDBFF'}
    } 
)
fig8.show()


# In[35]:


from dash.dependencies import Input, Output
import plotly.express as px

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}

app.layout = html.Div(style={'backgroundColor': colors['background']},children=[
    html.Div(style={'backgroundColor': colors['background']},children=
        html.H1(
        children='Heart Disease',
        style={
            'textAlign': 'center',
            'color': colors['text']
        }
        ),
            ),
    
    html.Div([
        html.Div([
            html.Div([
            html.H6(children='Total participants',
                style={
                    'textAlign':'center',
                    'color':'#FF4500'}       
            ),
            html.P('1025',
                  style={
                      'textAlign':'center',
                      'color':'black',
                      'fontsize':40}
                  )],className='card_container three rows',
                    style={'border-radius':'5px','background-color':'#DEB887','margin':'25px','position':'relative','box-shadow':'2px 2px 2px #DEB887'}),
            
            html.Div([
            html.H6(children='No. of Dieases',
                style={
                    'textAlign':'center',
                    'color':'#FF4500'}       
            ),
            html.P('526',
                  style={
                      'textAlign':'center',
                      'color':'black',
                      'fontsize':40}
                    ),
            html.P('52.5%',
                  style={
                      'textAlign':'center',
                      'color':'black',
                      'fontsize':40}
                  )],className='card_container three rows',
                    style={'border-radius':'5px','background-color':'#DEB887','margin':'25px','position':'relative','box-shadow':'2px 2px 2px #DEB887'}),
            html.Div([
            html.H6(children='No. of No-Dieases',
                style={
                    'textAlign':'center',
                    'color':'#FF4500'}       
            ),
            html.P('499',
                  style={
                      'textAlign':'center',
                      'color':'black',
                      'fontsize':40}
                    ),
            html.P('47.5%',
                  style={
                      'textAlign':'center',
                      'color':'black',
                      'fontsize':40}
                  )],className='card_container three rows',
                    style={'border-radius':'5px','background-color':'#DEB887','margin':'25px','position':'relative','box-shadow':'2px 2px 2px #DEB887'}),
            ],
            className='four columns'),
        html.Div([
            dcc.Graph(
                id='graph2',
                figure=fig2
            ),  
        ], className='four columns'),
        html.Div([
            html.Label(['']),
            dcc.Dropdown(
                id='my_dropdown',
                options=[
                    {'label': 'Gender', 'value': 'sex'},
                    {'label': 'Target', 'value': 'target'},
                    {'label': 'Chest pain', 'value': 'cp'},
                    {'label': 'blood suger(>120mmhg)', 'value': 'fbs'},
                    
                ],
                value='target',
                multi=False,
                clearable=False,
                style={"width": "100%"}
            ),
            dcc.Graph(id='Graph1'
                     ), 
        ], className='four columns'),
    ], className='row'),
    
    html.Div([
        html.Div([
            dcc.Graph(
                id='graph9',
                figure=fig3,
                    
            ),  
        ], className='six columns'),
        html.Div([
            dcc.Graph(
                id='graph4',
                figure=fig1
            ),  
        ], className='six columns'),
      
    ], className='row'),
    html.Div([
        html.Div([
            dcc.Graph(
                id='graph6',
                figure=fig5,
                    
            ),  
        ], className='four columns'),
        html.Div([
            html.P("x-axis:",style={'color':'#ff22ff'}),
            dcc.Checklist(
                id='x-axis',
                options=[{'value': x, 'label': x} 
                         for x in ['target']],
                value='target',
                style={'color':'#ffffff'},
                labelStyle={'display': 'inline-block'},
                
            ),
            html.P("y-axis:",style={'color':'#ff22ff'}),
            dcc.RadioItems(
                id='y-axis', 
                options=[{'value': x, 'label': x} 
                         for x in ['trestbps', 'chol']],
                value='chol',
                style={'color':'#ffffff'},
                labelStyle={'display': 'inline-block'}
            ),
            dcc.Graph(id="box-plot"),
        ], className='four columns'),
        html.Div([
            dcc.Graph(
                id='graph8',
                figure=fig8
                     ), 
        ], className='four columns'),
    ], className='row'),
    
])
@app.callback(
    Output(component_id='Graph1', component_property='figure'),
    [Input(component_id='my_dropdown', component_property='value')]
)

def update_graph(my_dropdown):
    dff = df

    piechart=px.pie(
            data_frame=dff,
            names=my_dropdown
            )
    piechart.update_layout(
    height = 400, width = 400)
    piechart.update_layout(
        {"plot_bgcolor":'#111111', "paper_bgcolor":'#111111','font': {'color': colors['text']}
        }
    )            
    return (piechart)
@app.callback(
    Output("box-plot", "figure"), 
    [Input("x-axis", "value"), 
     Input("y-axis", "value")])
def generate_chart(x, y):
    fig = px.box(df, x=x, y=y)
    
    fig.update_layout(
    {"plot_bgcolor":'#111111', "paper_bgcolor":'#111111','font': {'color': colors['text']}
    } 
)
    return fig
    


# In[ ]:


if __name__=='__main__':
    app.run_server()


# In[ ]:





# In[ ]:





# In[ ]:




