def main(path, name):
    # imports
    import speech_recognition as sr 
    import moviepy.editor as mp

    # SpaCy entity recognition
    import spacy
    nlp = spacy.load("en_core_web_sm")
    from collections import Counter

    # for personality analysis
    import pandas as pd
    import plotly.express as px
    import pickle
    import re


    ### video to text
    #take video
    clip = mp.VideoFileClip(path) 
    
    #convert to .wav
    clip.audio.write_audiofile(name + ".wav")

    #initialize recognizer and audio
    r = sr.Recognizer()
    audio = sr.AudioFile(name + ".wav")

    with audio as source:
        audio_file = r.record(source)
        result = r.recognize_google(audio_file)

    ### TODO figure out how to export it
    # exporting the result 
    with open(name + '.txt',mode ='w') as file: 
        file.write(result) 
    
    text = result

    ### entity recognition
    ent = nlp(text)  # nlp comes from spacy.load("en_core_web_sm") in import area

    tokens_lower = nlp(text.lower())

    #### TODO figure out how to make this customizable from frontend
    skill_keyword_list= ["seo", "market","research", "datas", "analytics", "contents", "developing","advertising","mobile"]
    exp_keyword_list = ["agency","years", "bachelors", "masters", "mba"]
    cult_keyword_list = ["collaboration","efficiency","hands on","learning","innovation","passion","excitement"]


    #remove stopwords and punctuations
    words = [token.text for token in tokens_lower if token.is_stop != True and token.is_punct != True]
    word_freq = Counter(words)


    # Calculate scores based on keywords
    skill_score = 0
    for word in skill_keyword_list:
        if word_freq[word] > 0:
            skill_score += 1

    

    cult_score = 0
    for word in cult_keyword_list:
        if word_freq[word] > 0:
            cult_score += 1
    

    exp_score = 0
    for word in exp_keyword_list:
        if word_freq[word] > 0:
            exp_score += 1
    

    ## print out the organizations that the algorithm recognized
    orgs = ""
    for X in ent.ents:
        if X.label_ == 'ORG':
            orgs += X.text + ", "
    




    ### personality analysis

    cEXT = pickle.load( open( "models/cEXT.p", "rb"))
    cNEU = pickle.load( open( "models/cNEU.p", "rb"))
    cAGR = pickle.load( open( "models/cAGR.p", "rb"))
    cCON = pickle.load( open( "models/cCON.p", "rb"))
    cOPN = pickle.load( open( "models/cOPN.p", "rb"))
    vectorizer_31 = pickle.load( open( "models/vectorizer_31.p", "rb"))
    vectorizer_30 = pickle.load( open( "models/vectorizer_30.p", "rb"))

    def predict_personality(text):
        scentences = re.split("(?<=[.!?]) +", text)
        text_vector_31 = vectorizer_31.transform(scentences)
        text_vector_30 = vectorizer_30.transform(scentences)
        EXT = cEXT.predict(text_vector_31)
        NEU = cNEU.predict(text_vector_30)
        AGR = cAGR.predict(text_vector_31)
        CON = cCON.predict(text_vector_31)
        OPN = cOPN.predict(text_vector_31)
        return [EXT[0], NEU[0], AGR[0], CON[0], OPN[0]]



    personality = predict_personality(text)
    
    df = pd.DataFrame(dict(r=personality, theta=['EXT','NEU','AGR', 'CON', 'OPN']))
    pers_fig = px.line_polar(df, r='r', theta='theta', line_close=True)
    ### TODO display fig on frontend?

    print("Skill score :", skill_score)
    print("Culture score :", cult_score)
    print("Experience score :", exp_score)
    print("Organizations recognized: ", orgs[:-2])
    print("Predicted personality:", personality)
    #fig.show()

    
    
    return skill_score, cult_score,exp_score,orgs[:-2],personality,pers_fig
    


if __name__ == "__main__":
    skill_score, cult_score,exp_score,orgs,personality,pers_fig = main("ConnorThompson.mp4", "ConnorThompson")
    
    
