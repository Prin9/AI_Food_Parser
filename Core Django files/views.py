from django.shortcuts import render

from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from tensorflow.keras.models import load_model
from recipe_scrapers import scrape_me

os.chdir('/home/blancliner/Desktop/Assignments/8th sem Major Proj/Colab_model_244')

#lastOutput={}

#creating a list of all the foods, in the argument i put the path to the folder that has all folders for food
def create_foodlist(path):
    list_ = list()
    for root, dirs, files in os.walk(path, topdown=False):
      for name in dirs:
        list_.append(name)
    return list_    

#loading the trained model  
my_model = load_model('model_trained (copy).h5', compile = False)
food_list = create_foodlist("../Local_model_200/food-101/images") 

#function to help in predicting classes of new images
def predict_class(img, show = True):
    model= my_model
    img = image.load_img(img, target_size=(224, 224 ))
    img = image.img_to_array(img)                    
    img = np.expand_dims(img, axis=0)         
    img /= 255.                                      

    pred = model.predict(img)
    index = np.argmax(pred)   
    food_list.sort()
    pred_value = food_list[index]
    pred_value=pred_value.replace('_',' ')
    print(pred_value,"\n\n==============")
    df = pd.read_csv('../Local_model_200/base_links.csv')
    scraper = scrape_me(df[df['title']==pred_value].values[0][1])
    #print(scraper.ingredients(),"\n\n",scraper.instructions(),"\n\n",scraper.instructions_list(),"\n\n",scraper.nutrients(),"\n\n\n\n")
    
    outDict={}
    outDict["text"]=""


    #Title
    outDict['text']+=pred_value+"\n"+ "================" + "\n\n"
    outDict['title']=pred_value

    #Ingredients
    outDict["text"]+="\n\nINGREDIENTS\n\n"
    outDict['ingredients']=''
    a = scraper.ingredients()
    for i in a:
        outDict["ingredients"]+=" -> "+i+'\n\n'
        outDict["text"]+=i+'\n'
    

    #Instructions

    outDict["text"]+="\n\nINSTRUCTIONS\n\n"
    outDict['instructions']=''
    a = scraper.instructions_list()
    for i in a:
        outDict['instructions']+=" -> "+i+'\n\n'
        outDict["text"]+=i+'\n'

    
    #Nutrition
    outDict["text"]+="\n\nNUTRITION VALUE\n\n"

    a= scraper.nutrients()
    print(a)
    outDict['calories'] = a['calories']
    outDict['text']+='Calories : ' + a['calories']+'\n'

    outDict['carbs'] = a['carbohydrateContent']
    outDict['text']+='carbs : ' + a['carbohydrateContent']+'\n'

    outDict['cholestrol'] = a['cholesterolContent']
    outDict['text']+='cholestrol : ' + a['cholesterolContent']+'\n'

    outDict['fiber'] = a['fiberContent']
    outDict['text']+='fiber : ' + a['fiberContent']+'\n'

    outDict['protein'] = a['proteinContent']
    outDict['text']+='protein : ' + a['proteinContent']+'\n'

    outDict['sodium'] = a['sodiumContent']
    outDict['text']+='sodium : ' + a['sodiumContent']+'\n'

    #outDict['sugar'] = a['sugarContent']
    #outDict['text']+='sugar : ' + a['sugarContent']+'\n'

    outDict['fat'] = a['fatContent']
    outDict['text']+='fat : ' + a['fatContent']+'\n'

    outDict['unsaturatedFats'] = a['unsaturatedFatContent']
    outDict['text']+='unsaturatedFats : ' + a['unsaturatedFatContent']+'\n'

    



    global lastOutput
    lastOutput = outDict

    
    ul = "\n"
    for i in scraper.ingredients():
        ul += "\n" + i + "\n"
    return outDict




# Create your views here.
def index(request):
    return render(request, "input.html")


def refresh(request):
    return render(request, "input.html")

def process(request):
    imgg = 'food-img' in request.FILES and request.FILES['food-img'] 
    imgg = request.POST.get('food-img',"NaN") #request.POST['food-img']
    output={}
    global lastOutput
    if imgg=="NaN":
        output = lastOutput
    else:
        output = predict_class(imgg)

    print("============>" , output)
    #print(type(imgg), "     !!!!!!!!!!!!!!!! ", imgg)
    
    return render(request, "result.html", {"title": output['title'], "imgPath": imgg, "ingredients":output["ingredients"], "instructions":output["instructions"], "calories":output["calories"], "carbs":output["carbs"], "cholestrol":output["cholestrol"], "fiber":output["fiber"], "protein":output["protein"], "sodium":output["sodium"], "fat":output["fat"], "unsaturatedFats":output["unsaturatedFats"], "text" : output["text"] })
