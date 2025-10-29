# Import Packages
import pandas as pd
import pickle
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1. Load and clean dataset
def get_clean_data():
    data = pd.read_csv('data.csv')
    data = data.drop(['Unnamed: 32', 'id'], axis=1)
    return data

# 2. Feature Engineering (custom features)
def create_custom_features(data):
    # Concavity per Fractal Dimension
    data['concavity_per_fractal_dimension_mean'] = data['concavity_mean'] / (data['fractal_dimension_mean'] + 1e-6)
    data['concavity_per_fractal_dimension_se'] = data['concavity_se'] / (data['fractal_dimension_se'] + 1e-6)
    data['concavity_per_fractal_dimension_worst'] = data['concavity_worst'] / (data['fractal_dimension_worst'] + 1e-6)
    
    # Compactness per Texture
    data['compactness_per_texture_mean'] = data['compactness_mean'] / (data['texture_mean'] + 1e-6)
    data['compactness_per_texture_se'] = data['compactness_se'] / (data['texture_se'] + 1e-6)
    data['compactness_per_texture_worst'] = data['compactness_worst'] / (data['texture_worst'] + 1e-6)
    
    # Concave Points per Perimeter
    data['concave_points_per_perimeter_mean'] = data['concave points_mean'] / (data['perimeter_mean'] + 1e-6)
    data['concave_points_per_perimeter_se'] = data['concave points_se'] / (data['perimeter_se'] + 1e-6)
    data['concave_points_per_perimeter_worst'] = data['concave points_worst'] / (data['perimeter_worst'] + 1e-6)
    
    # Compactness per Area
    data['compactness_per_area_mean'] = data['compactness_mean'] / (data['area_mean'] + 1e-6)
    data['compactness_per_area_se'] = data['compactness_se'] / (data['area_se'] + 1e-6)
    data['compactness_per_area_worst'] = data['compactness_worst'] / (data['area_worst'] + 1e-6)
    
    return data

# 3. Preprocessing + Transformation
def preprocess_data(data):
    # Encode target
    data['diagnosis'] = data['diagnosis'].apply(lambda x: 1 if x == 'M' else 0)
    
    # Create custom features
    data = create_custom_features(data)
    
    # Combine original + custom feature columns
    feature_cols = [c for c in data.columns if c != 'diagnosis']
    
    # PowerTransformer for all features
    pt = PowerTransformer(method='yeo-johnson')
    data[feature_cols] = pt.fit_transform(data[feature_cols])
    
    return data, pt, feature_cols

# 4. Train/Test Split + Scaling
def train_scale(data):
    X = data.drop(['diagnosis'], axis=1)
    y = data['diagnosis']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, scaler

# 5. Train Logistic Regression
def train_model(X_train, y_train, X_test, y_test):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    
    return model

# 6. Main function
def main():
    data = get_clean_data()
    data, pt, feature_cols = preprocess_data(data)
    X_train, X_test, y_train, y_test, scaler = train_scale(data)
    model = train_model(X_train, y_train, X_test, y_test)
    
    # Save model, scaler, and single PowerTransformer
    with open('model/model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('model/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open('model/power_transformer.pkl', 'wb') as f:
        pickle.dump(pt, f)
    with open('model/feature_cols.pkl', 'wb') as f:
        pickle.dump(feature_cols, f)
    
    print("Training completed. Model, scaler, transformer, and feature list saved.")

if __name__ == '__main__':
    main()
