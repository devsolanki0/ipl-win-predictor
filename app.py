from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
model_columns = pickle.load(open("columns.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    
    batting_team = request.form['batting_team']
    bowling_team = request.form['bowling_team']
    runs_to_get = int(request.form['runs_to_get'])
    balls_remaining = int(request.form['balls_remaining'])
    wickets = int(request.form['wickets'])
    current_score = int(request.form['current_score'])
    target = int(request.form['target'])

    input_df = pd.DataFrame(columns=model_columns)
    input_df.loc[0] = 0

    input_df['runs_to_get'] = runs_to_get
    input_df['balls_remaining'] = balls_remaining
    input_df['wickets'] = wickets
    input_df['current_score'] = current_score
    input_df['target'] = target

    bt_col = f'batting_team_{batting_team}'
    bw_col = f'bowling_team_{bowling_team}'

    if bt_col in input_df.columns:
        input_df[bt_col] = 1
    if bw_col in input_df.columns:
        input_df[bw_col] = 1

    probability = model.predict_proba(input_df)[0][1] * 100

    return render_template('index.html', prediction=round(probability,2))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)