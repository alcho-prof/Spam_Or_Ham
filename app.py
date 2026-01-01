from fastapi import FastAPI,Form
from fastapi.responses import HTMLResponse
from fastapi.responses import JSONResponse
import pickle

app=FastAPI() #create a FastAPI object 
#load the trained model and  tokenizer 

with open("models/feature_vectorizer.pkl","rb") as model_file:
    feature_extraction=pickle.load(model_file)

with open("models/logistic_regression.pkl","rb") as fe_file:
    model=pickle.load(fe_file)


#create a GET and POST request 

#initalize fastapi


@app.get("/",response_class=HTMLResponse) # default url
async def serve_html():
    with open("templates/index.html",'r') as file:
        return HTMLResponse(content=file.read()) 

#Creating a API for prdiction
@app.post('/predict')
async def predict_messgae(text:str=Form(...)):
    #(...) -> spread operators
    #preprocess the input text using tokenizer
    features=feature_extraction.transform([text])

    #predict using the model 
    prediction=model.predict(features)[0]

    #return the prediction result
    result="Ham" if prediction==1 else "Spam"

    return JSONResponse(content=result)