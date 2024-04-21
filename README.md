Unsupervised Multi-Topic Classification and Summarization with BART

```
pip install -r requirements.txt
```

Initial Classification (Clustering Algorithm)

For the unsupervised classification task we resort to clustering algorithms like GMM or Means Shift Clustering algorithms.

Reason being GMM and Mean Shift are better at classifying datapoints with an overlap than other algorithms such as Fuzzy K-Means.

1. Testing: 
- Pretrained model: [pretrained_50_epochs.pth](https://ssneduin-my.sharepoint.com/personal/mohit21110123_snuchennai_edu_in/_layouts/15/onedrive.aspx?ga=1&id=%2Fpersonal%2Fmohit21110123%5Fsnuchennai%5Fedu%5Fin%2FDocuments%2F6th%20Semester%2FProjects%2FNLP%20Project%2FBart%2Fpretrained)

- Place this model in ```./pretrained```
- Place your text in the variable ```text_input```
- Run the file ```python3 main.py```

