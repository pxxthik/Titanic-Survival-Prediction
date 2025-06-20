from helper import encode_sex, encode_embarked, encode_title, bin_fare, bin_age, is_alone
from services.model_loader import load_model

# Constants (could also be loaded from .env)
MODEL_NAME = "Titanic Survival Predictor"
DAGSHUB_URL = "https://dagshub.com"
REPO_OWNER = "pxxthik"
REPO_NAME = "Titanic-Survival-Prediction"

# Load model once at import
model = load_model(MODEL_NAME, DAGSHUB_URL, REPO_OWNER, REPO_NAME)


def transform_input(request):
    pclass = int(request.form['pclass'])
    sex = encode_sex(request.form['sex'])
    age = float(request.form['age'])
    fare = float(request.form['fare'])
    embarked = encode_embarked(request.form['embarked'])
    familysize = int(request.form['familysize'])
    title = encode_title(request.form['title'])
    fareBin = bin_fare(fare)
    ageBin = bin_age(age)
    isAlone = is_alone(familysize)

    return [pclass, sex, age, fare, embarked, familysize, isAlone, title, fareBin, ageBin]


def get_prediction(input_data):
    return model.predict([input_data])[0]
