from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import uvicorn



with open("aqi_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("label_encoder.pkl", "rb") as encoder_file:
    le = pickle.load(encoder_file)



app = FastAPI(
    title="AirSense AQI Prediction API",
    description="AQI Category Prediction using Machine Learning",
    version="1.0"
)



class AQIData(BaseModel):
    country: str
    city: str
    co_aqi_value: float
    ozone_aqi_value: float
    no2_aqi_value: float
    pm25_aqi_value: float




@app.get("/")
def home():
    return {
        "message": "AirSense AQI Prediction API Running Successfully"
    }




@app.post("/predict")
def predict(data: AQIData):

    input_data = [[
        data.co_aqi_value,
        data.ozone_aqi_value,
        data.no2_aqi_value,
        data.pm25_aqi_value
    ]]


    prediction = model.predict(input_data)


    predicted_category = le.inverse_transform(prediction)[0]

   
    statements = {
        "Good":
        "Air quality is considered satisfactory, and air pollution poses little or no health risk.",

        "Moderate":
        "Air quality is acceptable; however, some pollutants may pose a moderate health concern for sensitive individuals.",

        "Unhealthy for Sensitive Groups":
        "Sensitive groups may experience health effects. The general public is less likely to be affected.",

        "Unhealthy":
        "Air quality levels may begin to affect everyone, and sensitive groups could experience more serious health effects.",

        "Very Unhealthy":
        "Health alert: The risk of health effects is increased for everyone.",

        "Hazardous":
        "Air quality is extremely dangerous. Everyone is more likely to experience serious health effects."
    }


    statement = statements.get(
        predicted_category,
        "Air quality analysis completed successfully."
    )



    return {
        "country": data.country,
        "city": data.city,

        "aqi_prediction": predicted_category,

        "pollution_values": {
            "CO AQI Value": data.co_aqi_value,
            "Ozone AQI Value": data.ozone_aqi_value,
            "NO2 AQI Value": data.no2_aqi_value,
            "PM2.5 AQI Value": data.pm25_aqi_value
        },

        "analysis": statement
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)