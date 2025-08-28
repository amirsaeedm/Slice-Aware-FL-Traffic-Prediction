We followed below code and naming structure in this repository for our capstone work that is built on top of  [base paper](https://github.com/vperifan/Federated-Time-Series-Forecasting).

| NotebookNumber | MainCategory             | SubCategory                               | Description                                                                                       |
| ------ | ------------------------ | ----------------------------------------- | ------------------------------------------------------------------------------------------------- |
| 00     | base paper               | base paper LSTM                           | Load and run training of base paper LSTM on base data                                             |
| 01     | multistep LSTM           | basic multistep LSTM                      | Create, Train and evaluate first version of multistep LSTM                                        |
| 02     | Seq2Seq LSTM             | Seq2Seq LSTM base data                    | Create, Train and evaluate better version Sequence to Sequence multistep LSTM on base data        |
|        |                          | Seq2Seq LSTM base+extra data              | Create, Train and evaluate better version Sequence to Sequence multistep LSTM on base plus extra data |
| 03     | Clustering               | Cluster base data                         | Run the base data through unsupervized learning (Kmeans, DBScan, Heirarchical), visualize and save |
|        |                          | Cluster base+extra data                   | Run the base plus extra data through unsupervized learning (Kmeans, DBScan, Heirarchical), visualize and save |
| 04     | Seq2Seq LSTM Clustered   | Seq2Seq LSTM clustered base data          | Train and evaluate Sequence to Sequence multistep LSTM on clustered base data                      |
|        |                          | Seq2Seq LSTM clustered base+extra data    | Train and evaluate Sequence to Sequence multistep LSTM on clustered base plus extra data           |
| 05     | Transformer              | Transformer multistep base data           | Create, Train and evaluate Transformer multistep on base data                                     |
|        |                          | Transformer multistep clustered base data | Create, Train and evaluate Transformer multistep on clustered base data                           |
|        |                          | Transformer multistep clustered base+extra data | Create, Train and evaluate Transformer multistep on clustered base plus extra data            |
| 06     | Comparison               | Evaluation and Comparison base data       | Load and evaluate all previous models on base test dataset samples                                |
|        |                          | Evaluation and Comparison base+extra data | Load and evaluate all previous models on base plus extra test dataset samples                     |
|        |                          | Evaluation and Comparison base+extra inverse scaled | Load and evaluate all previous models on inverse scaled base plus extra test dataset samples |
| 07     | Process data             | Process extra data                        | Load, process and combine extra data with base paper                                              |
| 08     | Pipeline                 | Visualization pipeline                    | Create a pipeline for model, test dataset load and display on charts                              |
| 09     | Dashboard                | Streamlit Dashboard                       | Incorporate data and model pipeline into streamlit dashboard UI                                   |


We followed below `approach` and `architecture`:

![alt text](<FL architecture_capstone.png>)