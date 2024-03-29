import pandas as pd
import uvicorn
from fastapi import FastAPI
from Scripts import predict
from typing import List
from pydantic import BaseModel
import json

# from dumper import dump
app = FastAPI()


# @app.get('/')
# def index():
#     return {"Home": "Oolu churn prediction"}
#

class Account(BaseModel):
    account_id: int
    total_amount: int
    day_disabled: int
    expected_total_amount: int
    created_at: str
    total_payed: int
    status: int
    cutoff_days: int
    prev_payment: str


@app.post('/predictions')
def predictions(accounts: List[Account]):
    data = pd.DataFrame([account.__dict__ for account in accounts])
    df = data[['account_id']]
    df['churn_prob'] = predict.make_prediction(data)
    result = df.to_json(orient="records")
    parsed = json.loads(result)

    # dump(list(predict.make_prediction(data)))
    return {'payload': (json.dumps(parsed, indent=0))}


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)