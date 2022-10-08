# Imitation_strategy
研究用のリポジトリ  
  
# Usage
## ***edit_csv.py***  
csvファイルの編集用のスクリプト  
できることは以下の2つ  
- 2つのcsvファイルを1つにまとめる（-m, --merge） 
```python
python3 edit_csv.py -c1 {CSVFILE1_PATH} -c2 {CSVFILE2_PATH} -m
```  
  
- csvfile内の文字列を数字に置き換え、int型に変更する(-r, --revise)
```python
python3 edit_csv.py -c1 {CSVFILE1_PATH} -r
```  
  
## ***imitation.py***  
模倣学習（決定木の構築）を行うスクリプト  
```python
python3 imitation.py -r {ROLE_NAME}
```  
＊ csvファイルがcsvfilesディレクトリ内に ***文字列を数字に置き換え済み***の"ロール名.csv (ex. Attacker.csv)"があることが必要  
＊ modelsディレクトリ内に模倣モデルが生成される  
＊ graphディレクトリ内に構築した決定木(png)が保存される

## ***predict.py***
模倣したモデルを用いた予測を行うテスト用のスクリプト  
```python
python3 predit.py -r {ROLE_NAME}
```

# LICENSE
Copyright (c) 2022 Dan Sato
This repository is licensed under The MIT License, see [LICENSE](https://github.com/Dansato1203/Imitation_strategy/blob/main/LICENSE).
