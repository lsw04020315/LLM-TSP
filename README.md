Get Start
1.Install Python>= 3.8. For convenience, execute the following command.
pip install -r requirements.txt

2.Prepare Data. You can obtain the well pre-processed datasets from [Google Drive] or [Baidu Drive], 
Then place the downloaded data in the folder./dataset. Here is a summary of supported datasets.

3.Train and evaluate model. We provide the experiment scripts for all benchmarks under the folder ./scripts/. 
You can reproduce the experiment results as the following examples:
bash ./scripts/ETTh1.sh
bash ./scripts/ETTm1.sh
bash ./scripts/weather.sh
bash ./scripts/QBO.sh

If you are using Windows 10/11, you can also reproduce the issue by following these steps:

Step 1
Fill in the corresponding hyperparameters[ETTh1.sh,ETTm1.sh,weather.sh and QBO.sh]

Step 2
--task_name long_term_forecast --model_id MFFN-data-seq_len-pred_len

Step 3
run main.py



Develop your own model.
Add the model file to the folder ./models. You can follow the ./models/Transformer.py.
Include the newly added model in the Exp_Basic.model_dict of ./exp/exp_basic.py.
Create the corresponding scripts under the folder ./scripts.

Further Reading
Survey on Transformers in Time Series:

Qingsong Wen, Tian Zhou, Chaoli Zhang, Weiqi Chen, Ziqing Ma, Junchi Yan, and Liang Sun. 
"Transformers in time series: A survey.", IJCAI, 2023. [paper]

Acknowledgement
We appreciate the following github repos a lot for their valuable code base or datasets:

https://github.com/DAMO-DI-ML/ICML2022-FEDformer

https://github.com/thuml/Time-Series-Library

https://github.com/khegazy/One-Fits-All

https://github.com/DC-research/TEMPO