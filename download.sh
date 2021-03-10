###
 # @Description: 
 # @Autor: Jiachen Sun
 # @Date: 2021-03-09 18:17:42
 # @LastEditors: Jiachen Sun
 # @LastEditTime: 2021-03-09 19:48:17
### 
python gdrivedl.py https://drive.google.com/file/d/1vzmR0Yf0SF_EbK6DLxrpA2hNj0paB68G/view?usp=sharing
python gdrivedl.py https://drive.google.com/file/d/1iinRZzMqBsLr52R936Rcq7IlFl0cWd4P/view?usp=sharing

mkdir -p dataset
ln -s /cityspace/ dataset/cityscapes

mv train_epoch_200.pth ./exp/cityscapes/psanet50/model/
mv fine_val.txt  dataset/cityscapes/
