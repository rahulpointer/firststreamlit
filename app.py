import streamlit as st
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification

@st.cache(allow_output_mutation=True) #To make sure that the model is not downloaded again
def get_model_from_hub():
    tokenizer =  BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('rahulpointer/bertmodel_custom')
    return tokenizer,model

#Calling the method to get the tokenizer and the model.
tokenizer,model = get_model_from_hub()


text_data  = st.text_area('Enter the text')
button  = st.button('Submit for classification')

dictionary = {
    1:'Toxic',
    0:'Non-Toxic'
}

if text_data and button:
    token_rep = tokenizer([text_data],padding=True,truncation=True,max_length=512,return_tensors='pt')
    output = model(**token_rep)

    legits = output.legits
    prediction = np.argmax(output.legits.detach.numpy(),axis=-1)

    st.write('Legits',legits)
    st.write('predictions',dictionary[prediction])
    




