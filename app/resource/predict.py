
import aiohttp_jinja2
import jinja2
import numpy as np
import pandas as pd
from aiohttp import web
from app.common.utils import load_model
import pickle
import asyncio

# Create a dictionary mapping for State values
state_mapping = {'New York': 0, 'California': 1, 'Florida': 2}

def predict_profit(data):
    # Load the saved model
    model = load_model()

    profit_prediction = model.predict(data)

    # Return the predicted profit values
    return profit_prediction

class PredictView(web.View):
    @aiohttp_jinja2.template('index.html')
    async def get(self):
        return {}

    @aiohttp_jinja2.template('predict.html')
    async def post(self):
        input_values = await self.request.post()
        rd_spend = float(input_values['rd_spend'])
        administration = float(input_values['administration'])
        marketing_spend = float(input_values['marketing_spend'])
        state = state_mapping[input_values['state']]

        input_array = np.zeros((1, 4))
        input_array[0][0] = rd_spend
        input_array[0][1] = administration
        input_array[0][2] = marketing_spend
        input_array[0][3] = state

        prediction = predict_profit(input_array)[0]

        return {'prediction': round(prediction, 2)}






