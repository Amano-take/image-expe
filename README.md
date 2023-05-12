# image-expe  
Assig  
課題のフォルダーが入っているページです  
Layer  
各種レイヤープラス様々な処理が入っています  
Advanced  
  +- 3D  
     カラー画像処理の際に使おうと思っています。  
  +- CNN.py  
     コンテストのパラメータ獲得に使いました  
  基本的に発展課題Aのためのフォルダです
AdvancedB
このフォルダ内のみ外部ライブラリKeras, TensorFlowを用いています。
  +- Gan
      GANを用いた画像生成のプログラムです。TensorFlowを用いています。
  +- Image
      ステップごとの出力画像です。result_10400.pngが最もうまくいった出力です。
Contest  
  +- forme  
     簡易的な正答率チェック, そのための答え作成pythonファイル, 答えtxtが入っています  
  +- CNN.py, le4MNIST_X.txt, predict.txt  
     CNN.pyで取得したパラーメーターをもとに、predict.txtを作成しています.  
Parameters  
コンテストのためのパラーメーターが入っています  
今回の提出ではcontest_finを使用しました.  

Layer
  モデルの作成に使用したものです。numpyのみのimportにとどめています。

コンテストについて  
始め200個でテストしてパラーメーターの優劣を決定しているので、
厳格な学習とはなっていません.時間があれば独立にパラーメーターを作成したいと思います.  

使ったモデルについて  
一層目にアーギュメンテーション, 畳み込みとプーリング  
二層目にバッチノーマライゼーション, relu関数, Dropout  
三層目にSoftmax, CrossEntropy関数  
それぞれの間には全結合を行っています  
-> 全結合を一つだけにしました. 
-> さらに小さなdata-argumentationを行い過学習させた物をfinetuningパラメータとして,  
それを適切なdata-argumentationを模索しながら,パラメータ更新を行っていきました。

厳密な層数の定義がわからないので,三層ではないかもしれません.  

アーギュメントの種類について  
Affin変換（回転, 平行移動, スキュー...), noiseその他(ガウシアンノイズ, cutout, randomcrop)を追加しています  
オリジナルとして, contestdateに数字とは関係ない白点を含むデータを見たので,そのようなwhitenoise(?)も追加しています.  
Layer内RandomArgumentation.pyにてこの作業を行っています。

ハイパラについて
畳み込みch 16 フィルター幅5
プーリング 4*4
Dropout 0.5
中間層数 400
バッチ数 100
エポック数 100


