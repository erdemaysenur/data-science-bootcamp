import gradio as gr
import torch
import transformers

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
        self.bert = transformers.AutoModelForSequenceClassification.from_pretrained(self.MODEL_NAME)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.MODEL_NAME)
        #self.linear = torch.nn.Linear(self.bert.config.hidden_size, 2)
        #self.softmax = torch.nn.Softmax(dim=1)
        
    def forward(self, input_text):
        encoding = self.tokenizer.encode_plus(
            input_text,
            add_special_tokens = True, 
            pad_to_max_length = True,
            return_token_type_ids = False,
            return_attention_mask = True,
            return_tensors = 'pt' 
        )

        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)

        output = self.bert(
            input_ids = input_ids,
            attention_mask = attention_mask)

        return output


def predict(input_text):
    model = Model()
    model.load_state_dict(torch.load("model.pth", map_location=device), strict=False)
    model.eval()
    outputs = model(input_text)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=-1)
    if prediction.item() == 0:
        return "NEGATIVE"
    if prediction.item() == 1:
        return "POSITIVE"


iface = gr.Interface(predict,
inputs="text",
outputs="text",
title="Bert Base Sentiment Analysis",
description="This is a bert based sentiment classifier that is trained with tinder application reviews (EN).",
allow_flagging="never")
iface.launch(inbrowser=True)
