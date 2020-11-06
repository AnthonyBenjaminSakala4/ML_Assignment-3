import streamlit as st
import pickle
import numpy as np

# kmeans=pickle.load(open('unsupervisedmodels.pkl','rb')) 


def predict_kmeans(CountryName,StringencyLegacyIndexForDisplay,StringencyIndexForDisplay,	StringencyIndex,StringencyLegacyIndex,ContainmentHealthIndexForDisplay,ContainmentHealthIndex,GovernmentResponseIndexForDisplay,ConfirmedCases,ConfirmedDeaths,EconomicSupportIndexForDisplay,E2_Debtcontractrelief,EconomicSupportIndex,C3_Cancelpublicevents,C1_Schoolclosing):
    input=np.array([[CountryName,StringencyLegacyIndexForDisplay,StringencyIndexForDisplay,	StringencyIndex,StringencyLegacyIndex,ContainmentHealthIndexForDisplay,ContainmentHealthIndex,GovernmentResponseIndexForDisplay,ConfirmedCases,ConfirmedDeaths,EconomicSupportIndexForDisplay,E2_Debtcontractrelief,EconomicSupportIndex,C3_Cancelpublicevents,C1_Schoolclosing]]).astype(np.float64)
    prediction=kmeans.predict(input)
    return prediction

def main():
    st.title("Records of countries classified in the clusters")
    html_temp = """
    <div style="background-color:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;">Unsupervised ML App </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    CountryName = st.text_input("CountryName","Type Here",key='0')
    StringencyLegacyIndexForDisplay = st.text_input("StringencyLegacyIndexForDisplay","Type Here",key='1')
    StringencyIndexForDisplay = st.text_input("StringencyIndexForDisplay","Type Here",key='2')
    StringencyIndex = st.text_input("StringencyIndex","Type Here",key='3')
    StringencyLegacyIndex = st.text_input("StringencyLegacyIndex","Type Here",key='4')
    ContainmentHealthIndexForDisplay = st.text_input("ContainmentHealthIndexForDisplay","Type Here",key='5')
    GovernmentResponseIndexForDisplay = st.text_input("GovernmentResponseIndexForDisplay","Type Here",key='6')
    ContainmentHealthIndex = st.text_input("ContainmentHealthIndex","Type Here",key='7')
    ConfirmedCases = st.text_input("ConfirmedCases","Type Here",key='8')
    ConfirmedDeaths = st.text_input("ConfirmedDeaths","Type Here",key='9')
    EconomicSupportIndexForDisplay = st.text_input("EconomicSupportIndexForDisplay","Type Here",key='9')
    E2_Debtcontractrelief = st.text_input("E2_Debtcontractrelief","Type Here",key='10')
    EconomicSupportIndex = st.text_input("EconomicSupportIndex","Type Here",key='11')
    C3_Cancelpublicevents = st.text_input("C3_Cancelpublicevents","Type Here",key='12')
    C1_Schoolclosing = st.text_input("C1_Schoolclosing","Type Here",key='13')

    safe_html="""  
      <div style="background-color:#F4D03F;padding:10px >
       <h2 style="color:white;text-align:center;"> Your forest is safe</h2>
       </div>
    """
    danger_html="""  
      <div style="background-color:#F08080;padding:10px >
       <h2 style="color:black ;text-align:center;"> Your forest is in danger</h2>
       </div>
    """

    if st.button("Predict"):
        output=predict_kmeans(CountryName,StringencyLegacyIndexForDisplay,StringencyIndexForDisplay,	StringencyIndex,StringencyLegacyIndex,ContainmentHealthIndexForDisplay,ContainmentHealthIndex,GovernmentResponseIndexForDisplay,ConfirmedCases,ConfirmedDeaths,EconomicSupportIndexForDisplay,E2_Debtcontractrelief,EconomicSupportIndex,C3_Cancelpublicevents,C1_Schoolclosing)
        st.success('This country located in this cluster {}'.format(output))


if __name__=='__main__':
    main()
