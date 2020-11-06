# %%writefile app.py

features = new_dataset[['CountryName','StringencyLegacyIndexForDisplay','StringencyIndexForDisplay',	'StringencyIndex','StringencyLegacyIndex',
              'ContainmentHealthIndexForDisplay','ContainmentHealthIndex','GovernmentResponseIndexForDisplay','ConfirmedCases','ConfirmedDeaths',
              'EconomicSupportIndexForDisplay','E2_Debt/contract relief','EconomicSupportIndex','C3_Cancel public events','C1_School closing']]
select = VarianceThreshold()
features_selected = select.fit_transform(features)

# Commented out IPython magic to ensure Python compatibility.
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
import streamlit as st

model = KMeans(n_clusters = 6)

pca = PCA(n_components=2).fit(features_selected)
pca_2d = pca.transform(features_selected)

model.fit(pca_2d)

labels = model.predict(pca_2d)

first = pca_2d[:, 0]
second = pca_2d[:, 1]
plt.scatter(first, second, c = labels)
plt.scatter(model.cluster_centers_[:,0],model.cluster_centers_[:,1],color='purple',marker='*',label='centroid')

kmeans = KMeans(n_clusters=10)
kmeans.fit(df_first_half)
plt.scatter(df_first_half[:,0],df_first_half[:,1], c=kmeans.labels_, cmap='rainbow')

range_n_clusters = [2, 3, 4, 5, 6]

#km.cluster_centers_

from sklearn import datasets
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
transformed = scaler.fit_transform(x)
# Plotting 2d t-Sne
x_axis = transformed[:,0]
y_axis = transformed[:,1]

kmeans = KMeans(n_clusters=4, random_state=42,n_jobs=-1)
y_pred =kmeans.fit_predict(transformed)

predicted_label = kmeans.predict([[7,7.2, 3.5, 0.8, 1.6,7.2, 3.5, 0.8, 1.6,7.2, 3.5, 0.8, 1.67, 7.2, 3.5]])
predicted_label

import streamlit as st
import pickle
import numpy as np

# kmeans=pickle.load(open('unsupervisedmodels.pkl','rb')) 


def predict_kmeans(CountryName,StringencyLegacyIndexForDisplay,StringencyIndexForDisplay,	StringencyIndex,StringencyLegacyIndex,ContainmentHealthIndexForDisplay,ContainmentHealthIndex,GovernmentResponseIndexForDisplay,ConfirmedCases,ConfirmedDeaths,EconomicSupportIndexForDisplay,E2_Debtcontractrelief,EconomicSupportIndex,C3_Cancelpublicevents,C1_Schoolclosing):
    input=np.array([[CountryName,StringencyLegacyIndexForDisplay,StringencyIndexForDisplay,	StringencyIndex,StringencyLegacyIndex,ContainmentHealthIndexForDisplay,ContainmentHealthIndex,GovernmentResponseIndexForDisplay,ConfirmedCases,ConfirmedDeaths,EconomicSupportIndexForDisplay,E2_Debtcontractrelief,EconomicSupportIndex,C3_Cancelpublicevents,C1_Schoolclosing]]).astype(np.float64)
    prediction=kmeans.predict(input)
    return prediction

def main():
    st.title("Classifying Countries in clusters")
    html_temp = """
    <div style="background-color:#520236 ;padding:10px">
    <h2 style="color:white;text-align:center;">Unsupervised ML App </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    CountryName = st.text_input("what is the CountryName?","Type Here",key='0')
    StringencyLegacyIndexForDisplay = st.text_input("what is the StringencyLegacyIndexForDisplay?","Type Here",key='1')
    StringencyIndexForDisplay = st.text_input("what is the StringencyIndexForDisplay?","Type Here",key='2')
    StringencyIndex = st.text_input("what is the StringencyIndex?","Type Here",key='3')
    StringencyLegacyIndex = st.text_input("what is the StringencyLegacyIndex?","Type Here",key='4')
    ContainmentHealthIndexForDisplay = st.text_input("what is the ContainmentHealthIndexForDisplay?","Type Here",key='5')
    GovernmentResponseIndexForDisplay = st.text_input("what is the GovernmentResponseIndexForDisplay?","Type Here",key='6')
    ContainmentHealthIndex = st.text_input("what is the ContainmentHealthIndex?","Type Here",key='7')
    ConfirmedCases = st.text_input("what is the ConfirmedCases?","Type Here",key='8')
    ConfirmedDeaths = st.text_input("what is the ConfirmedDeaths?","Type Here",key='9')
    EconomicSupportIndexForDisplay = st.text_input("what is the EconomicSupportIndexForDisplay?","Type Here",key='9')
    E2_Debtcontractrelief = st.text_input("what is the E2_Debtcontractrelief?","Type Here",key='10')
    EconomicSupportIndex = st.text_input("what is the EconomicSupportIndex?","Type Here",key='11')
    C3_Cancelpublicevents = st.text_input("what is the C3_Cancelpublicevents?","Type Here",key='12')
    C1_Schoolclosing = st.text_input("what is the C1_Schoolclosing?","Type Here",key='13')

    safe_html="""  
      <div style="background-color:#F4D03F;padding:10px >
       <h2 style="color:white;text-align:center;"> Your country is safe</h2>
       </div>
    """
    danger_html="""  
      <div style="background-color:#F08080;padding:10px >
       <h2 style="color:black ;text-align:center;"> Your country is in danger</h2>
       </div>
    """

    if st.button("Predict"):
        output=predict_kmeans(CountryName,StringencyLegacyIndexForDisplay,StringencyIndexForDisplay,	StringencyIndex,StringencyLegacyIndex,ContainmentHealthIndexForDisplay,ContainmentHealthIndex,GovernmentResponseIndexForDisplay,ConfirmedCases,ConfirmedDeaths,EconomicSupportIndexForDisplay,E2_Debtcontractrelief,EconomicSupportIndex,C3_Cancelpublicevents,C1_Schoolclosing)
        st.success('This country is located in this cluster {}'.format(output))


if __name__=='__main__':
    main()
