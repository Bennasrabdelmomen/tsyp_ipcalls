import torch
import re
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from transformers import BertTokenizer, BertForQuestionAnswering

sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
qa_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
qa_model = BertForQuestionAnswering.from_pretrained("bert-base-uncased")

cve_data = pd.read_csv('cve_data.csv')
mitre_data = pd.read_csv('mitre_data.csv')

cve_embeddings = torch.load("cve_embeddings.pt", map_location=torch.device("cpu"))
mitre_embeddings = torch.load("mitre_embeddings.pt", map_location=torch.device("cpu"))

def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    return text

def get_similar_entries(query, embeddings, data, top_k=5):
    query_embedding = sbert_model.encode(preprocess_text(query), convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(query_embedding, embeddings).cpu()
    top_results = torch.topk(cosine_scores, k=top_k)

    results = []
    for score, idx in zip(top_results[0].tolist(), top_results[1].tolist()):
        result = {
            'entry': data.iloc[idx].to_dict(),
            'score': score
        }
        results.append(result)
    return results

def answer_question(question, context):
    inputs = qa_tokenizer(question, context, return_tensors="pt")
    answer_start_scores, answer_end_scores = qa_model(**inputs).values()

    answer_start = torch.argmax(answer_start_scores)
    answer_end = torch.argmax(answer_end_scores) + 1
    answer = qa_tokenizer.convert_tokens_to_string(
        qa_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end])
    )
    return answer

def cybersecurity_chatbot(query):
    similar_cves = get_similar_entries(query, cve_embeddings, cve_data)
    similar_mitres = get_similar_entries(query, mitre_embeddings, mitre_data)

    cve_results = [
        {
            "CVE-ID": entry['entry'].get('CVE-ID', 'N/A'),
            "Description": entry['entry'].get('DESCRIPTION', 'N/A'),
            "Severity": entry['entry'].get('SEVERITY', 'N/A'),
            "Score": entry['score']
        }
        for entry in similar_cves
    ]

    mitre_results = [
        {
            "Question": entry['entry'].get('Question', 'N/A'),
            "Answer": entry['entry'].get('Answer', 'N/A'),
            "Score": entry['score']
        }
        for entry in similar_mitres
    ]

    return cve_results, mitre_results
