import csv
from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def display_predictions():
    predictions = []
    with open('predictions.csv', 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            predictions.append({
                'company': row['Company'],
                'prediction': row['Prediction']
            })
    return render_template('index.html', predictions=predictions)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)
