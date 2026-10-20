import joblib
import pandas as pd

def predict_price(str):
    input_dict = {
        'hectares': float(str['outdoorSpace']),
        'bath': int(str['numBathrooms']),
        'bed': int(str['numRooms']),
        'house_size': float(str['size']),
        'state': int(str['location']),
        }
    print(input_dict)
    # Carrega o modelo a partir do ficheiro
    XGB_from_joblib = joblib.load('modeloXGB.pkl')
    result = XGB_from_joblib.predict(pd.DataFrame(input_dict, index=[0]))  # Aqui são carregados os inputs
    return result

# %%
