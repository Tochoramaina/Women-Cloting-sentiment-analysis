##Women's Clothing E-commerce

Sentiment prediction and Topic Modelling
In this repository i implement a dual-stage naural language processing (NLP) pipeline to decode a customer review on a cloth. I combined Bidirectional LSTM and topic extraction to transform raw text into structured, actionable business intelligence 

##Pipeline Artchitecture 
###stage 1
In the first stage, I used deep learning to predict a review as 'Bad', 'Neutral' or 'Good' and then driiled down on negative reviews by using LDA to find root causes of customers negative reviews.
*Model. I used Bidirectional LSTM so as it catches the context of a word based on the word before or behind it.
*Attention layers. I built an attention block(using tanh and softmax) to force the model to give weights to important terms rather than give all words an eqaul importance score.
*Handling overfitting. I implemented L2 regularizers and Drop-outs to make the model robust thus avoiding a model that just memorizes the training data.
*Output. The model made predictions and confidence scores i.e probabilities of why a review is 'neutral', 'bad' or 'good'. 
####Model prediction results
|Review | Prediction | Probability Score|
|
| :--- | :--- | :--- |
| **Review: ** This dress is absolutely stunning and fits perfectly |  Good | 98.79% |
| **Review: ** The fabric felt cheap and it arrived with a hole in the sleeve | Bad | 51.92% |
| **Review: ** It is a basic t-shirt.Nothing special, but itdoes the job | Good | 78.63% |
| **Review: ** I wanted to love this, but the sizing is completely off |  Bad | 42.77% |
 |


### stage 2 
I implemented LDA(Latent Dirichlet Allocation) to extract topics/words that made customer's reviews negative and these were the findings:
####Topic modelling results 
|Topic | Key Terms | Qualitative Analysis |
|
|:--- | :--- | :---|
| **Topic 0** | Sweater, fabric, returned, color | **Quality and Description Gap:**Clothes were not of the expected quality or color, prompting customer returns. |
| **Topic 1** | material, fabric, cut, flattering | **Aesthetics & Fit: ** Focuses on how a cloth is made and how it looks/fits after being worn. |
| **Topic 2** | disappointed, retailer, small, zipper, ordered | **Fuctional and Sizing Issues:** Physical defects (zipper) and items nor matching the size ordered. |
|

> **Developer Note:** I observed there was overlaps between topic0 and topic 1 even if I adjusted bigrams min_counts and threshold for filtering phrases but I couldn't decouple quality and aesthetics from this dataset customers feedback.

 
