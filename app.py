import pandas as pd
import numpy as np
from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

@app.route("/")
def home():
    return render_template('index.html')

@app.route('/model/',methods=['GET', 'POST'])
def model():
    if request.method == 'POST':
        capShape = request.form["cap-shape"],         
        capSurface = request.form["cap-surface"],            
        capColor = request.form["cap-color"],
        bruises = request.form["bruises"],      
        odor = request.form["odor"],            
        gillAttachment = request.form["gill-attachment"],        
        gillSpacing = request.form["gill-spacing"],           
        gillSize = request.form["gill-size"],             
        gillColor = request.form["gill-color"],             
        stalkShape = request.form["stalk-shape"],            
        stalkRoot = request.form["stalk-root"],            
        stalkSurfaceAboveRing = request.form["stalk-surface-above-ring"],
        stalkSurfaceBelowRing = request.form["stalk-surface-below-ring"],
        stalkColorAboveRing = request.form["stalk-color-above-ring"],
        stalkColorBelowRing = request.form["stalk-color-below-ring"],
        veilType = request.form["veil-type"],             
        veilColor = request.form["veil-color"],            
        ringNumber = request.form["ring-number"],             
        ringType = request.form["ring-type"],            
        sporePrintColor = request.form["spore-print-color"],      
        population = request.form["population"],          
        habitat = request.form["habitat"]

        df_list = [
            capShape[0], 
            capSurface[0], 
            capColor[0], 
            bruises[0], 
            odor[0], 
            gillAttachment[0], 
            gillSpacing[0], 
            gillSize[0], 
            gillColor[0], 
            stalkShape[0], 
            stalkRoot[0], 
            stalkSurfaceAboveRing[0], 
            stalkSurfaceBelowRing[0], 
            stalkColorAboveRing[0], 
            stalkColorBelowRing[0], 
            veilType[0], 
            veilColor[0], 
            ringNumber[0], 
            ringType[0], 
            sporePrintColor[0], 
            population[0], 
            habitat[0]
        ]

        df_arr = np.array(df_list)
        df_arr = df_arr.reshape(1,22)
        df = pd.DataFrame(df_arr)

        model = joblib.load('models\model.joblib')
        pred = model.predict(df)
        
        if pred[0] == 0:
            pred='Edible'
        elif pred[0]==1:
            pred = 'Poisonous'

        return render_template('prediction.html', pred=pred)
    
    else:
        return render_template('model.html')

if __name__=="__main__":
    app.run()