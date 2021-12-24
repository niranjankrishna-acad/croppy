from types import new_class
from transformers import DetrFeatureExtractor, DetrForSegmentation
from PIL import Image
import gradio as gr
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import torch
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


feature_extractor = DetrFeatureExtractor.from_pretrained('facebook/detr-resnet-101-panoptic')
model = DetrForSegmentation.from_pretrained('facebook/detr-resnet-101-panoptic')

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
sentenceModel = AutoModel.from_pretrained('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

def classSimilarity(query,probs,bboxes,img):
    new_classes = []
    for i in range(len(probs)):
        new_classes.append(model.config.id2label[probs[i].argmax().item()])
    new_classes.append(query)
    # Tokenize sentences
    encoded_input = tokenizer(new_classes, padding=True, truncation=True, return_tensors='pt')

    # Compute token embeddings
    with torch.no_grad():
        model_output = sentenceModel(**encoded_input)

    # Perform pooling. In this case, max pooling.
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    cosineScores = []
    for i in range(0, len(sentence_embeddings) - 1):
        cosineScores.append(cosine_similarity(sentence_embeddings[-1].reshape(1, -1),sentence_embeddings[i].reshape(1, -1))[0][0])
    maxIndex = cosineScores.index(max(cosineScores))
    bound = bboxes[maxIndex].tolist()
    w,h = img.size
    xmin, ymin, xmax, ymax = bound[0], bound[1], bound[2], bound[3]
    return img.crop((xmin, ymin, xmax, ymax))
def croppy(image,query):
    classes = []
    values = list(model.config.id2label.values())
    count = 0 
    for i in range(len(values)):
        if "LABEL" not in values[i]:
            count+=1
            classes.append(values[i])
    
   
    image = Image.fromarray(np.uint8(image))
    inputs = feature_extractor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    probas = outputs.logits.softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > 0.9
    target_sizes = torch.tensor(image.size[::-1]).unsqueeze(0)
    postprocessed_outputs = feature_extractor.post_process(outputs, target_sizes)
    bboxes_scaled = postprocessed_outputs[0]['boxes'][keep]
    img = classSimilarity(query, probas[keep], bboxes_scaled,image)
    return img
gr.Interface(fn=croppy,inputs=["image","text"],outputs = "image").launch(server_name='0.0.0.0')