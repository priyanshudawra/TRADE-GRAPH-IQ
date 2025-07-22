from flask import Flask, render_template, request, jsonify
from model import train_and_evaluate_model, predict_new_data

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train():
    stock_name = request.form['stock_name']
    mse, rmse, r2, svm_model, scaler, graph_url = train_and_evaluate_model(stock_name)

    # Store model and scaler in a global dictionary (in-memory for simplicity)
    global trained_models
    trained_models[stock_name] = {'model': svm_model, 'scaler': scaler}

    return jsonify({
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'message': f'Model trained for {stock_name}',
        'graph': graph_url  # Include the graph URL in the response
    })

@app.route('/predict', methods=['POST'])
def predict():
    stock_name = request.form['stock_name']
    if stock_name not in trained_models:
        return jsonify({'error': 'Model not trained for this stock'}), 400

    svm_model = trained_models[stock_name]['model']
    scaler = trained_models[stock_name]['scaler']
    predictions = predict_new_data(stock_name, svm_model, scaler)
    return jsonify({'predictions': predictions})

if __name__ == '__main__':
    trained_models = {}
    app.run(debug=True)
