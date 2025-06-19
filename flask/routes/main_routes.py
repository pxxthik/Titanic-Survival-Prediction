from flask import Blueprint, render_template, request
from services.prediction_service import transform_input, get_prediction

main_blueprint = Blueprint('main', __name__)

@main_blueprint.route('/')
def home():
    return render_template('index.html', prediction=None)

@main_blueprint.route('/predict', methods=['POST'])
def predict():
    input_data = transform_input(request)
    prediction = get_prediction(input_data)
    return render_template('index.html', prediction=prediction)
