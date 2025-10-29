#Import the package
import streamlit as st
import pickle
import pandas as pd
import plotly.graph_objects as go
import numpy as np

#Getting the features data:
def get_clean_data():
    data = pd.read_csv('data.csv')
    data = data.drop(['Unnamed: 32'], axis = 1)
    data = data.drop(['id'], axis = 1)
    return data

#Function to create streamlit sidebar
def add_sidebar():
    st.sidebar.header("Cell Nuclei Details")
    data = get_clean_data()
    slider_labels = [        
        ("Radius (mean)", "radius_mean"),
        ("Texture (mean)", "texture_mean"),
        ("Perimeter (mean)", "perimeter_mean"),
        ("Area (mean)", "area_mean"),
        ("Smoothness (mean)", "smoothness_mean"),
        ("Compactness (mean)", "compactness_mean"),
        ("Concavity (mean)", "concavity_mean"),
        ("Concave Points (mean)", "concave points_mean"),
        ("Symmetry (mean)", "symmetry_mean"),
        ("Fractal Dimension (mean)", "fractal_dimension_mean"),
        ("Radius (SE)", "radius_se"),
        ("Texture (SE)", "texture_se"),
        ("Perimeter (SE)", "perimeter_se"),
        ("Area (SE)", "area_se"),
        ("Smoothness (SE)", "smoothness_se"),
        ("Compactness (SE)", "compactness_se"),
        ("Concavity (SE)", "concavity_se"),
        ("Concave Points (SE)", "concave points_se"),   
        ("Symmetry (SE)", "symmetry_se"),
        ("Fractal Dimension (SE)", "fractal_dimension_se"),
        ("Radius (worst)", "radius_worst"),
        ("Texture (worst)", "texture_worst"),
        ("Perimeter (worst)", "perimeter_worst"),
        ("Area (worst)", "area_worst"),
        ("Smoothness (worst)", "smoothness_worst"),
        ("Compactness (worst)", "compactness_worst"),
        ("Concavity (worst)", "concavity_worst"),
        ("Concave Points (worst)", "concave points_worst"),
        ("Symmetry (worst)", "symmetry_worst"),
        ("Fractal Dimension (worst)", "fractal_dimension_worst"),
    ]
    #Dictinary or object with key value pair
    input_dict = {}
    #Loop through each labels to create the slider
    for label, key in slider_labels:
        input_dict[key] = st.sidebar.slider(label, min_value = float(0), max_value = float(data[key].max()), value = float(data[key].mean()))
    #Return The Dictinary
    return input_dict

#Function to get the scaled values for radar_chart
def get_scaled_values(input_data):
    data = get_clean_data()
    X = data.drop(['diagnosis'], axis=1)
    scaled_dict = {}
    for key, value in input_data.items():
        value = input_data[key]
        min_val = X[key].min()
        max_val = X[key].max()
        scaled_dict[key] = (value - min_val) / (max_val - min_val)
    return scaled_dict

#Function to create the radar chart using plotly
def get_radar_chart(input_data):
    input_data = get_scaled_values(input_data)
    categories = [
        'Radius', 'Texture', 'Perimeter', 'Area', 'Smoothness',
        'Compactness', 'Concavity', 'Concave Points', 'Symmetry', 'Fractal Dimension'
    ]

    mean_keys = [
        'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean'
    ]

    se_keys = [k.replace('_mean', '_se') for k in mean_keys]

    worst_keys = [k.replace('_mean', '_worst') for k in mean_keys]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
      r=[input_data[k] for k in mean_keys],
      theta=categories,
      fill='toself',
      name='Mean Value'
    ))
    fig.add_trace(go.Scatterpolar(
      r=[input_data[key] for key in se_keys],
      theta=categories,
      fill='toself',
      name='Standard Error'
    ))
    fig.add_trace(go.Scatterpolar(
      r=[input_data[key] for key in worst_keys],
      theta=categories,
      fill='toself',
      name='Worst Value'
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
            visible=True,
            range=[0, 1]
            )
        ),
        showlegend=True
    )

    return fig

#Function to add the prediction
def add_predictions(input_data):
    # Load model, scaler, and single power transformer
    with open('model/model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('model/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('model/power_transformer.pkl', 'rb') as f:
        pt = pickle.load(f)
    with open('model/feature_cols.pkl', 'rb') as f:
        feature_cols = pickle.load(f)  # exact feature order

    # Ensure all custom features are present

    # 1️⃣ Concavity per Fractal Dimension
    input_data['concavity_per_fractal_dimension_mean'] = input_data['concavity_mean'] / (input_data['fractal_dimension_mean'])
    input_data['concavity_per_fractal_dimension_se'] = input_data['concavity_se'] / (input_data['fractal_dimension_se'])
    input_data['concavity_per_fractal_dimension_worst'] = input_data['concavity_worst'] / (input_data['fractal_dimension_worst'])

    # 2️⃣ Compactness per Texture
    input_data['compactness_per_texture_mean'] = input_data['compactness_mean'] / (input_data['texture_mean'])
    input_data['compactness_per_texture_se'] = input_data['compactness_se'] / (input_data['texture_se'])
    input_data['compactness_per_texture_worst'] = input_data['compactness_worst'] / (input_data['texture_worst'])

    # 3️⃣ Concave Points per Perimeter
    input_data['concave_points_per_perimeter_mean'] = input_data['concave points_mean'] / (input_data['perimeter_mean'])
    input_data['concave_points_per_perimeter_se'] = input_data['concave points_se'] / (input_data['perimeter_se'])
    input_data['concave_points_per_perimeter_worst'] = input_data['concave points_worst'] / (input_data['perimeter_worst'])
    
    # 4️⃣ Compactness per Area
    input_data['compactness_per_area_mean'] = input_data['compactness_mean'] / (input_data['area_mean'])
    input_data['compactness_per_area_se'] = input_data['compactness_se'] / (input_data['area_se'])
    input_data['compactness_per_area_worst'] = input_data['compactness_worst'] / (input_data['area_worst'])

    # Arrange values in the same order as training features
    all_features = np.array([input_data[f] for f in feature_cols]).reshape(1, -1)

    # Transform using single PowerTransformer
    all_features_transformed = pt.transform(all_features)

    # Scale features
    all_scaled = scaler.transform(all_features_transformed)

    # Predict
    prediction = model.predict(all_scaled)
    prediction_proba = model.predict_proba(all_scaled)

    # Sub Header
    st.subheader("Cell cluster prediction")
    st.write("The cell cluster is:")

    # Display
    if prediction[0] == 0:
        st.write("<span class= 'diagnosis-benign'>Benign</span>", unsafe_allow_html=True)
    else:
        st.write("<span class = 'diagnosis-malignant'>Malignant</span>", unsafe_allow_html=True)
    
    st.write("Prediction of the cells being Benign: \n", prediction_proba[0][0])
    st.write("Prediction of the cells being Malignant: \n", prediction_proba[0][1])

    st.write("This application is designed to support medical professionals in the diagnostic process. However, it should not replace a formal medical diagnosis or professional clinical judgment.")


def main():
    # My page configuration in streamlit.
    st.set_page_config(
        page_title="Breast Cancer Predictor",
        page_icon="👩🏻‍⚕️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    #Adding the css
    with open("assets/style.css") as f:
        st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)
    #Creating the sidebar
    input_data = add_sidebar()
    #My Container or div
    with st.container():
        st.title("Breast Cancer Predictor")
        st.write("Please connect this app to your cytology lab to help diagnose breast cancer form your tissue sample. This app predicts using a machine learning model whether a breast mass is benign or malignant based on the measurements it receives from your cytosis lab. You can also update the measurements by hand using the sliders in the sidebar.")
    # Creating the columns
    col1, col2 = st.columns([4,1])
    #First Column:
    with col1:
        radar_chart = get_radar_chart(input_data)
        st.plotly_chart(radar_chart)
    #Second Column:    
    with col2:
        add_predictions(input_data)

if __name__ == '__main__':
    main()