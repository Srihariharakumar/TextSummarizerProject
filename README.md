"# AI_Summary" 
Let's go through each code cell step by step:

### Code Cell 1:
```python
from pprint import pprint
from rouge_score import rouge_scorer
```
This cell imports the pprint function from the pprint module and the RougeScorer class from the rouge_score module. pprint is used for pretty printing data structures, while RougeScorer is used for computing ROUGE scores for text summarization evaluation.

### Code Cell 2:
```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
```
This cell imports the AutoTokenizer and AutoModelForSeq2SeqLM classes from the transformers library. It then initializes a tokenizer and a model using the BART-Large-CNN pre-trained model from the Hugging Face model hub.

### Code Cell 3:
```python
sample_text = '''
The majority of available text summarization datasets include short-form source documents that lack long-range causal and temporal dependencies, and often contain strong layout and stylistic biases. While relevant, such datasets will offer limited challenges for future generations of text summarization systems. We address these issues by introducing BookSum, a collection of datasets for long-form narrative summarization. Our dataset covers source documents from the literature domain, such as novels, plays and stories, and includes highly abstractive, human written summaries on three levels of granularity of increasing difficulty: paragraph-, chapter-, and book-level. The domain and structure of our dataset poses a unique set of challenges for summarization systems, which include: processing very long documents, non-trivial causal and temporal dependencies, and rich discourse structures. To facilitate future work, we trained and evaluated multiple extractive and abstractive summarization models as baselines for our dataset.
'''
```
This cell defines a sample text for summarization.

### Code Cell 4:
```python
X_token = tokenizer(sample_text, return_tensors="pt")['input_ids']
X_token
```
This cell tokenizes the sample text using the previously initialized tokenizer and returns the input token IDs as a PyTorch tensor.

### Code Cell 5:
```python
output_tensor = model.generate(X_token)

output = tokenizer.decode(output_tensor[0], skip_special_tokens=True)
```
This cell generates a summary using the pre-trained BART model by passing the tokenized input to the `generate` method. It then decodes the output tensor to obtain the generated summary as a string.

### Code Cell 6:
```python
pprint(output)
```
This cell prints the generated summary using the pprint function for a better formatted output.

### Code Cell 7:
```python
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer=True)
```
This cell initializes a RougeScorer object with specified metrics ('rouge1', 'rouge2', 'rougeLsum') and enables stemming.

### Code Cell 8:
```python
scores = scorer.score(sample_text, output)
print("ROUGE-1 (Unigram):", scores['rouge1'])
print("ROUGE-2 (Bigram):", scores['rouge2'])
print("ROUGE-L (Longest Common Subsequence):", scores['rougeLsum'])
```
This cell computes ROUGE scores for the generated summary by comparing it to the sample text and prints out the results for ROUGE-1, ROUGE-2, and ROUGE-L metrics.

The rest of the code cells follow a similar pattern, but they involve PEFT (Positional Encoding Free Transformer) model instead of the BART model for comparison purposes. The process involves defining the PEFT model, generating a summary using it, evaluating the generated summary using ROUGE metrics, and printing the results. Finally, it saves the PEFT model for future use.
