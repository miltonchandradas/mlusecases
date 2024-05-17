import os
import pickle
import numpy as np
from flask import Flask, render_template, request, redirect

# Creates Flask serving engine
app = Flask(__name__)

model = None
appHasRunBefore = False
flower = ""
sepal_length = ""
sepal_width = ""
petal_length = ""
petal_width = ""


@app.before_request
def init():
    """
    Load model else crash, deployment will not start
    """
    global model
    global appHasRunBefore

    if not appHasRunBefore:
        # Load the model...
        model = pickle.load(open('my_model.pkl', 'rb'))
        appHasRunBefore = True
        return None


@app.route("/v2/greet", methods=["GET"])
def status():
    global model
    if model is None:
        return "Flask Code: Model was not loaded."
    else:
        return "Flask Code: Model was loaded successfully..."


@app.route("/v2/predict", methods=["POST"])
def predict():
    global model
    global age
    global sex
    global bmi
    global bp
    global s1
    global s2
    global s3
    global s4
    global s5
    global s6

    if model is None:
        return "Flask Code: Model was not loaded."
    else:
        query = dict(request.json)
        age = query["age"]
        sex = query["sex"]
        bmi = query["bmi"]
        bp = query["bp"]
        s1 = query["s1"]
        s2 = query["s2"]
        s3 = query["s3"]
        s4 = query["s4"]
        s5 = query["s5"]
        s6 = query["s6"]
        attributes = [age, sex, bmi, bp, s1, s2, s3, s4, s5, s6]
        prediction = model.predict(
            # (trailing comma) <,> to make batch with 1 observation
            np.array([list(map(float, attributes)),])
        )

        return {"attributes": {"age": age, "sex": sex, "bmi": bmi, "bp": bp, "s1": s1, "s2": s2, "s3": s3, "s4": s4, "s5": s5, "s6": s6 }, "Prediction": str(prediction)}


if __name__ == "__main__":
    print("Serving Initializing")
    init()
    print("Serving Started")
    app.run(debug=True)
