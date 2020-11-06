
import pickle
#pickle.dump(kmeans,open('unsupervisedmodels.pkl','wb'))
import streamlit as st
import pickle
# import numpy as np
# from sklearn.cluster import KMeans
# #kmeans=pickle.load(open('unsupervisedmodels.pkl','rb')) 
# from sklearn import datasets
# from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# transformed = scaler.fit_transform(x)
# # Plotting 2d t-Sne
# x_axis = transformed[:,0]
# y_axis = transformed[:,1]

# kmeans = KMeans(n_clusters=4, random_state=42,n_jobs=-1)
# #y_pred =kmeans.fit_predict(transformed)

# # def predict_kmeans(CountryName,StringencyLegacyIndexForDisplay,StringencyIndexForDisplay,	StringencyIndex,StringencyLegacyIndex,ContainmentHealthIndexForDisplay,ContainmentHealthIndex,GovernmentResponseIndexForDisplay,ConfirmedCases,ConfirmedDeaths,EconomicSupportIndexForDisplay,E2_Debtcontractrelief,EconomicSupportIndex,C3_Cancelpublicevents,C1_Schoolclosing):
# #     input=np.array([[CountryName,StringencyLegacyIndexForDisplay,StringencyIndexForDisplay,	StringencyIndex,StringencyLegacyIndex,ContainmentHealthIndexForDisplay,ContainmentHealthIndex,GovernmentResponseIndexForDisplay,ConfirmedCases,ConfirmedDeaths,EconomicSupportIndexForDisplay,E2_Debtcontractrelief,EconomicSupportIndex,C3_Cancelpublicevents,C1_Schoolclosing]]).astype(np.float64)
# #     prediction=kmeans.predict(input)
# #     return prediction

# st.title("Records of countries classified in the clusters")
# html_temp = """
# <div style="background-color:#025246 ;padding:12px">
# <h2 style="color:white;text-align:center;">Unsupervised App </h2>
# </div>
# """
# st.markdown(html_temp, unsafe_allow_html=True)
# CountryName = st.text_input("CountryName","Type Here",key='0')
# StringencyLegacyIndexForDisplay = st.text_input("StringencyLegacyIndexForDisplay","Type Here",key='1')
# StringencyIndexForDisplay = st.text_input("StringencyIndexForDisplay","Type Here",key='2')
# StringencyIndex = st.text_input("StringencyIndex","Type Here",key='3')
# StringencyLegacyIndex = st.text_input("StringencyLegacyIndex","Type Here",key='4')
# ContainmentHealthIndexForDisplay = st.text_input("ContainmentHealthIndexForDisplay","Type Here",key='5')
# GovernmentResponseIndexForDisplay = st.text_input("GovernmentResponseIndexForDisplay","Type Here",key='6')
# ContainmentHealthIndex = st.text_input("ContainmentHealthIndex","Type Here",key='7')
# ConfirmedCases = st.text_input("ConfirmedCases","Type Here",key='8')
# ConfirmedDeaths = st.text_input("ConfirmedDeaths","Type Here",key='9')
# EconomicSupportIndexForDisplay = st.text_input("EconomicSupportIndexForDisplay","Type Here",key='9')
# E2_Debtcontractrelief = st.text_input("E2_Debtcontractrelief","Type Here",key='10')
# EconomicSupportIndex = st.text_input("EconomicSupportIndex","Type Here",key='11')
# C3_Cancelpublicevents = st.text_input("C3_Cancelpublicevents","Type Here",key='12')
# C1_Schoolclosing = st.text_input("C1_Schoolclosing","Type Here",key='13')

# if st.button("Predict"):
#   output=predict_kmeans(CountryName,StringencyLegacyIndexForDisplay,StringencyIndexForDisplay,	StringencyIndex,StringencyLegacyIndex,ContainmentHealthIndexForDisplay,ContainmentHealthIndex,GovernmentResponseIndexForDisplay,ConfirmedCases,ConfirmedDeaths,EconomicSupportIndexForDisplay,E2_Debtcontractrelief,EconomicSupportIndex,C3_Cancelpublicevents,C1_Schoolclosing)
#   st.success('This country located in this cluster {}'.format(output))

# -*- coding: utf-8 -*-
"""Assignment3.ipynb
"""

import pandas as pd

data= pd.read_csv('https://raw.githubusercontent.com/Diane10/ML/master/assignment3.csv')
# data.info()

# data.isnull().sum()

null_counts = data.isnull().sum().sort_values()
selected = null_counts[null_counts < 8000 ]

percentage = 100 * data.isnull().sum() / len(data)


data_types = data.dtypes
# data_types

missing_values_table = pd.concat([null_counts, percentage, data_types], axis=1)
# missing_values_table

col=['CountryName','Date','StringencyLegacyIndexForDisplay','StringencyIndexForDisplay','ContainmentHealthIndexForDisplay','GovernmentResponseIndexForDisplay',
'EconomicSupportIndexForDisplay','C8_International travel controls','C1_School closing','C3_Cancel public events','C2_Workplace closing','C4_Restrictions on gatherings',
'C6_Stay at home requirements','C7_Restrictions on internal movement','H1_Public information campaigns','E1_Income support','C5_Close public transport','E2_Debt/contract relief','StringencyLegacyIndex','H3_Contact tracing','StringencyIndex','ContainmentHealthIndex','E4_International support','EconomicSupportIndex','E3_Fiscal measures','H5_Investment in vaccines','ConfirmedCases','ConfirmedDeaths']

newdataset=data[col]
newdataset= newdataset.dropna()

from sklearn.preprocessing import LabelEncoder
newdataset['CountryName']=LabelEncoder().fit_transform(newdataset['CountryName'])


# # map features to their absolute correlation values
# corr = newdataset.corr().abs()

# # set equality (self correlation) as zero
# corr[corr == 1] = 0

# # of each feature, find the max correlation
# # and sort the resulting array in ascending order
# corr_cols = corr.max().sort_values(ascending=False)

# # display the highly correlated features
# display(corr_cols[corr_cols > 0.9])

# len(newdataset)

X=newdataset[['CountryName','StringencyLegacyIndexForDisplay','StringencyIndexForDisplay',	'StringencyIndex','StringencyLegacyIndex','ContainmentHealthIndexForDisplay','ContainmentHealthIndex','GovernmentResponseIndexForDisplay','ConfirmedCases','ConfirmedDeaths','EconomicSupportIndexForDisplay','E2_Debt/contract relief','EconomicSupportIndex','C3_Cancel public events','C1_School closing']]
# X=newdataset[['CountryName','StringencyLegacyIndexForDisplay','StringencyIndexForDisplay',	'StringencyIndex','StringencyLegacyIndex','ContainmentHealthIndexForDisplay','ContainmentHealthIndex','GovernmentResponseIndexForDisplay','ConfirmedCases','ConfirmedDeaths']]

# df_first_half = X[:1000]
# df_second_half = X[1000:]

# """Feature selector that removes all low-variance features."""

from sklearn.feature_selection import VarianceThreshold

selector = VarianceThreshold()
x= selector.fit_transform(X)

df_first_half = x[:5000]
df_second_half = x[5000:]

# """Create clusters/classes of similar records using features selected in (1),  use an unsupervised learning algorithm of your choice."""

# Commented out IPython magic to ensure Python compatibility.
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
import streamlit as st

# wcss=[]
# for i in range(1,11):
#     kmeans=KMeans(n_clusters=i, init='k-means++',random_state=0)
#     kmeans.fit(x)
#     wcss.append(kmeans.inertia_)
# st.set_option('deprecation.showPyplotGlobalUse', False)
# plt.plot(range(1,11),wcss)
# plt.title('The Elbow Method')
# plt.xlabel('Number of Clusters')
# plt.ylabel('WCSS')
# plt.show()
# st.pyplot()

model = KMeans(n_clusters = 6)

pca = PCA(n_components=2).fit(x)
pca_2d = pca.transform(x)

model.fit(pca_2d)

labels = model.predict(pca_2d)
# labels
# predicted_label = model.predict([[7.2, 3.5, 0.8, 1.6]])

# pca = PCA(n_components=2).fit(df_first_half)
# pca_2d = pca.transform(df_first_half)
# pca_2d

xs = pca_2d[:, 0]
ys = pca_2d[:, 1]
plt.scatter(xs, ys, c = labels)
plt.scatter(model.cluster_centers_[:,0],model.cluster_centers_[:,1],color='purple',marker='*',label='centroid')

kmeans = KMeans(n_clusters=10)
kmeans.fit(df_first_half)
plt.scatter(df_first_half[:,0],df_first_half[:,1], c=kmeans.labels_, cmap='rainbow')

range_n_clusters = [2, 3, 4, 5, 6]

# from sklearn.metrics import silhouette_samples, silhouette_score
# import matplotlib.cm as cm
# import numpy as np

# for n_clusters in range_n_clusters:
#     # Create a subplot with 1 row and 2 columns
#     fig, (ax1, ax2) = plt.subplots(1, 2)
#     fig.set_size_inches(18, 7)

#     # The 1st subplot is the silhouette plot
#     # The silhouette coefficient can range from -1, 1 but in this example all
#     # lie within [-0.1, 1]
#     ax1.set_xlim([-0.1, 1])
#     # The (n_clusters+1)*10 is for inserting blank space between silhouette
#     # plots of individual clusters, to demarcate them clearly.
#     ax1.set_ylim([0, len(pca_2d) + (n_clusters + 1) * 10])

#     # Initialize the clusterer with n_clusters value and a random generator
#     # seed of 10 for reproducibility.
#     clusterer = KMeans(n_clusters=n_clusters, random_state=10)
#     cluster_labels = clusterer.fit_predict(pca_2d)

#     # The silhouette_score gives the average value for all the samples.
#     # This gives a perspective into the density and separation of the formed
#     # clusters
#     silhouette_avg = silhouette_score(pca_2d, cluster_labels)
#     print("For n_clusters =", n_clusters,
#           "The average silhouette_score is :", silhouette_avg)

#     # Compute the silhouette scores for each sample
#     sample_silhouette_values = silhouette_samples(pca_2d, cluster_labels)
#     y_lower = 10
#     for i in range(n_clusters):
#         # Aggregate the silhouette scores for samples belonging to
#         # cluster i, and sort them
#         ith_cluster_silhouette_values = \
#             sample_silhouette_values[cluster_labels == i]

#         ith_cluster_silhouette_values.sort()

#         size_cluster_i = ith_cluster_silhouette_values.shape[0]
#         y_upper = y_lower + size_cluster_i

#         color = cm.nipy_spectral(float(i) / n_clusters)
#         ax1.fill_betweenx(np.arange(y_lower, y_upper),
#                           0, ith_cluster_silhouette_values,
#                           facecolor=color, edgecolor=color, alpha=0.7)

#         # Label the silhouette plots with their cluster numbers at the middle
#         ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

#         # Compute the new y_lower for next plot
#         y_lower = y_upper + 10  # 10 for the 0 samples

#     ax1.set_title("The silhouette plot for the various clusters.")
#     ax1.set_xlabel("The silhouette coefficient values")
#     ax1.set_ylabel("Cluster label")

#     # The vertical line for average silhouette score of all the values
#     ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

#     ax1.set_yticks([])  # Clear the yaxis labels / ticks
#     ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

#     # 2nd Plot showing the actual clusters formed
#     colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
#     ax2.scatter( pca_2d[:, 0], pca_2d[:, 1], marker='.', s=30, lw=0, alpha=0.7,
#                 c=colors, edgecolor='k')
#     # Labeling the clusters
#     centers = clusterer.cluster_centers_
#     # Draw white circles at cluster centers
#     ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
#                 c="white", alpha=1, s=200, edgecolor='k')

#     for i, c in enumerate(centers):
#         ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
#                     s=50, edgecolor='k')

#     ax2.set_title("The visualization of the clustered data.")
#     ax2.set_xlabel("Feature space for the 1st feature")
#     ax2.set_ylabel("Feature space for the 2nd feature")

#     plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
#                   "with n_clusters = %d" % n_clusters),
#                  fontsize=14, fontweight='bold')
# plt.show()

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

# from sklearn.manifold import TSNE
# tsne = TSNE(random_state=17)

# X_tsne = tsne.fit_transform(transformed)

# plt.figure(figsize=(12,10))
# plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_pred, 
#             edgecolor='none', alpha=0.7, s=40,
#             cmap=plt.cm.get_cmap('nipy_spectral', 10))
# plt.colorbar()
# plt.title('cluster. t-SNE projection');

# pca = PCA(n_components=2)
# X_reduced = pca.fit_transform(transformed)

# print('Projecting %d-dimensional data to 2D' % X.shape[1])

# plt.figure(figsize=(12,10))
# plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y_pred, 
#             edgecolor='none', alpha=0.7, s=40,
#             cmap=plt.cm.get_cmap('nipy_spectral', 10))
# plt.colorbar()
# plt.title('cluster. PCA projection');
# st.pyplot()

# """https://www.kaggle.com/kashnitsky/topic-7-unsupervised-learning-pca-and-clustering"""

# import seaborn as sns

# import pickle
# pickle.dump(kmeans,open('unsupervisedmodels.pkl','wb'))

# """Create a platform where new records of countries can be classified in the clusters"""



# Commented out IPython magic to ensure Python compatibility.
# %%writefile app.py
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



