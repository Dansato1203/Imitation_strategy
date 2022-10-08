# Imitation_strategy
研究用のリポジトリ  
  
## Usage
1. ***edit_csv.py***
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
  
2. ***imitation.py***
模倣学習（決定木の構築）を行うスクリプト  
```python
python3 imitation.py -r {ROLE_NAME}
```  
＊ csvファイルがcsvfilesディレクトリ内に "ロール名.csv (ex. Attacker.csv)"があることが必要  
＊ modelsディレクトリ内に模倣モデルが生成される
＊ graphディレクトリ内に構築した決定木(png)が保存される

3. ***predict.py***
模倣したモデルを用いた予測を行うテスト用のスクリプト  
```python
python3 predit.py -r {ROLE_NAME}
```

