from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.http import JsonResponse
import joblib  # Used to load a pre-trained model (example for scikit-learn)
import numpy as np
import csv
from django.views.decorators.http import require_POST
# from .models import Location, WeatherData
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.linear_model import SGDRegressor


# Load your trained model (assuming you have already trained and saved it)
# model = joblib.load("./ml_models/best_model.pkl")

@api_view(['POST'])
def predict_view(request):
    # try:
    #     # Extracting data from the request
    #     data = request.data
    #     aerosel_height = float(data.get('aerosel_height', 0))
    #     temperature = float(data.get('temperature', 0))
    #     dew = float(data.get('dew', 0))
    #     humidity = float(data.get('humidity', 0))
    #     precipitation = float(data.get('precipitation', 0))
    #     windspeed = float(data.get('windspeed', 0))
    #     sea_level_pressure = float(data.get('sea_level_pressure', 0))
    #     uv_index = float(data.get('uv_index', 0))

    #     # Form the input for the model (ensure the right input format)
    #     features = np.array([[aerosel_height, temperature, dew, humidity, precipitation, windspeed, sea_level_pressure, uv_index]])

    #     # Make prediction
    #     prediction = model.predict(features)
    #     result = prediction[0]  # Assuming model returns a single prediction

    #     return JsonResponse({"prediction": result}, status=200)

    # except Exception as e:
    #     return JsonResponse({"error": str(e)}, status=400)
    
    data = request.data
    ID = data.get('locationId')
    if  ID =='lubbock':
        df=pd.read_csv('./ml_models/LubbocklandfillML.csv')
    elif ID =='big-spring':
        df=pd.read_csv('./ml_models/BigSpringML.csv')
    elif ID==  'enlink-plant':
        df=pd.read_csv('./ml_models/EnlinkPlantML.csv')
    elif ID == 'midland-school-district':
        df = pd.read_csv('./ml_models/MidlandIndependentSchoolDistrictML.csv')
    elif  ID == '14401-fm-3503-midland':
        df = pd.read_csv('./ml_models/14401FM3503MidlandML.csv')
    elif  ID == 'etc-gas-fieldpipeline':
        df = pd.read_csv('./ml_models/EtcGasFieldpipelineML.csv')
    elif   ID == 'kinetik-pecos-plant':
        df = pd.read_csv('./ml_models/KinetikPecosPlantML.csv')
    elif   ID == '9201-farm-rd':
        df = pd.read_csv('./ml_models/9201FarmRdML.csv')
    elif    ID == 'van-horn-texas':
        df = pd.read_csv('./ml_models/VanHornTexasML.csv')
    elif ID == 'reagan-county-gas-fields':
        df = pd.read_csv('./ml_models/ReaganCountygasfeildsML.csv')
    
    X_features=['aerosol_height (m)', 'temp (°C)','dew (°C)','humidity (%)','precip (mm)',
            'windspeed (km/h)','sealevelpressure (mb)','UVindex']  #feature column
    y_target='CH4(ppb)'  #target column
    X_train=df[X_features].values
    y_train=df[y_target].values
    #Normalizing the training data
    norm_scale=StandardScaler()
    X_norm=norm_scale.fit_transform(X_train)
    #regression model
    reg_mod=SGDRegressor(max_iter=10000)
    reg_mod.fit(X_norm,y_train)
    #finding and printing the weight and bias
    w_normalize=reg_mod.coef_
    b_normalize=reg_mod.intercept_
    #trying to see if prediction is matching with the target values
    y_pred_sgd=reg_mod.predict(X_norm)
    y_pred=np.dot(X_norm,w_normalize)+ b_normalize
    joblib.dump(reg_mod,'best_model.pkl')
    loaded_model=joblib.load('best_model.pkl')
    data = request.data
    user_input = np.array([[
    float(data.get('aerosolHeight',3000)),
    float(data.get('temperature')),
    float(data.get('dew')),
    float(data.get('humidity') ),
    float(data.get('precipitation') ),
    float(data.get('windSpeed') ),
    float(data.get('seaLevelPressure') ),
    float(data.get('uvIndex') )
    ]])
    user_input_norm=norm_scale.transform(user_input)
    future_prediction=loaded_model.predict(user_input_norm) 
    future_prediction = future_prediction.tolist()
    return JsonResponse({"prediction": future_prediction}, status=200)
