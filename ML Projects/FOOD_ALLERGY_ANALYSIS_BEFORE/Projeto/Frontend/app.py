from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "GET":
        return render_template("index.html")
    else:
        GENDER_FACTOR_ENCODER = request.form["GENDER_FACTOR_ENCODER"]
        RACE_FACTOR_ENCODER = request.form["RACE_FACTOR_ENCODER"]
        ETHNICITY_FACTOR_ENCODER = request.form["ETHNICITY_FACTOR_ENCODER"]
        ATOPIC_MARCH_COHORT_ENCODER = request.form["ATOPIC_MARCH_COHORT_ENCODER"]
        SHELLFISH_ALG_START = request.form["SHELLFISH_ALG"]
        SHELLFISH_ALG_END = request.form["SHELLFISH_ALG"]
        FISH_ALG_START = request.form["FISH_ALG"]
        FISH_ALG_END = request.form["FISH_ALG"]
        MILK_ALG_START = request.form["MILK_ALG"]
        MILK_ALG_END = request.form["MILK_ALG"]
        SOY_ALG_START = request.form["SOY_ALG"]
        SOY_ALG_END = request.form["SOY_ALG"]
        EGG_ALG_START = request.form["EGG_ALG"]
        EGG_ALG_END = request.form["EGG_ALG"]
        WHEAT_ALG_START = request.form["WHEAT_ALG"]
        WHEAT_ALG_END = request.form["WHEAT_ALG"]
        PEANUT_ALG_START = request.form["PEANUT_ALG"]
        PEANUT_ALG_END = request.form["PEANUT_ALG"]
        SESAME_ALG_START = request.form["SESAME_ALG"]
        SESAME_ALG_END = request.form["SESAME_ALG"]
        WALNUT_ALG_START = request.form["WALNUT_ALG"]
        WALNUT_ALG_END = request.form["WALNUT_ALG"]
        ALMOND_ALG_START = request.form["ALMOND_ALG"]
        ALMOND_ALG_END = request.form["ALMOND_ALG"]
        CASHEW_ALG_START = request.form["CASHEW_ALG"]
        CASHEW_ALG_END = request.form["CASHEW_ALG"]
        ATOPIC_DERM_START = request.form["ATOPIC_DERM"]
        ATOPIC_DERM_END = request.form["ATOPIC_DERM"]
        ALLERGIC_RHINITIS_START = request.form["ALLERGIC_RHINITIS"]
        ALLERGIC_RHINITIS_END = request.form["ALLERGIC_RHINITIS"]
        ASTHMA_START = request.form["ASTHMA"]
        ASTHMA_END = request.form["ASTHMA"]

        try:
            columns = ['GENDER_FACTOR_ENCODER', 'RACE_FACTOR_ENCODER', 'ETHNICITY_FACTOR_ENCODER',
                       'PAYER_FACTOR_ENCODER', 'ATOPIC_MARCH_COHORT_ENCODER', 'AGE_START_YEARS',
                       'SHELLFISH_ALG_START', 'SHELLFISH_ALG_END', 'FISH_ALG_START', 'FISH_ALG_END', 'MILK_ALG_START',
                       'MILK_ALG_END', 'SOY_ALG_START', 'SOY_ALG_END', 'EGG_ALG_START', 'EGG_ALG_END',
                                       'WHEAT_ALG_START', 'WHEAT_ALG_END', 'PEANUT_ALG_START', 'PEANUT_ALG_END', 'SESAME_ALG_START',
                                       'SESAME_ALG_END', 'WALNUT_ALG_START', 'WALNUT_ALG_END', 'ALMOND_ALG_START',
                                       'ALMOND_ALG_END', 'CASHEW_ALG_START', 'CASHEW_ALG_END', 'ATOPIC_DERM_START',
                                       'ATOPIC_DERM_END', 'ALLERGIC_RHINITIS_START', 'ALLERGIC_RHINITIS_END', 'ASTHMA_START',
                                       'ASTHMA_END']
            df = pd.DataFrame(columns=columns,
                              data=np.array([GENDER_FACTOR_ENCODER, RACE_FACTOR_ENCODER,
                                             ETHNICITY_FACTOR_ENCODER, 0, ATOPIC_MARCH_COHORT_ENCODER, 0,
                                             SHELLFISH_ALG_START, SHELLFISH_ALG_END, FISH_ALG_START, FISH_ALG_END, MILK_ALG_START,
                                             MILK_ALG_END, SOY_ALG_START, SOY_ALG_END, EGG_ALG_START, EGG_ALG_END,
                                             WHEAT_ALG_START, WHEAT_ALG_END, PEANUT_ALG_START, PEANUT_ALG_END, SESAME_ALG_START,
                                             SESAME_ALG_END, WALNUT_ALG_START, WALNUT_ALG_END, ALMOND_ALG_START,
                                             ALMOND_ALG_END, CASHEW_ALG_START, CASHEW_ALG_END, ATOPIC_DERM_START,
                                             ATOPIC_DERM_END, ALLERGIC_RHINITIS_START, ALLERGIC_RHINITIS_END, ASTHMA_START,
                                             ASTHMA_END]).reshape(1, len(columns)))

            model = joblib.load('./model/random_forest_model.sav')
            if model.predict(df) == 1:
                return render_template("alergic.html")
            else:
                return render_template("not_alergic.html")

        except Exception as e:
            print(e)
            return "Algo falhor, verifique os dados!"


if __name__ == "__main__":
    app.run()
