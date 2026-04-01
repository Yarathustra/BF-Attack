你能为我的论文库BF-Attack写一个About和README文档（用英文）吗，内容分别是

1 基于开源仓库https://github.com/Trustworthy-AI-Group/TransferAttack/tree/main?tab=readme-ov-file

进行修改 ,他的协议是MIT License 

2 环境配置：conda create -n BF python=3.9 -y
conda activate BF
pip install git+https://github.com/openai/CLIP.git
pip install -r requirements.txt 

3 生成扰动样本使用
bash attack.sh
验证扰动样本ASR使用
bash validation.sh

4 我的实验结果：
The attack success rates (\%) of black-box attacks against six normally trained models. The adversarial examples are crafted via Inc-v3, Inc-v4, IncRes-v2 and Res-152.
此处插入figure

5 关于文章的citation，还在投Information Fusion还没有出预印本。

