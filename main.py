import csv
from flask import Flask, render_template, send_file

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

# Route to serve the Python code file
@app.route('/view_code')
def view_code():
    # Specify the path to the Python code file
    code_file_path = 'predictioncode.py'
    # Read the contents of the Python code file
    with open(code_file_path, 'r') as file:
        code_content = file.read()
    # Render the code content within the HTML template
    return render_template('code_view.html', code_content=code_content)


if __name__ == "__main__":
    app.run(debug=True)
