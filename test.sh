###
 # @Description: 
 # @Autor: Jiachen Sun
 # @Date: 2021-03-09 18:38:31
 # @LastEditors: Jiachen Sun
 # @LastEditTime: 2021-03-09 18:44:12
### 

PYTHONPATH=./ python tool/demo.py --config=config/cityscapes/cityscapes_psanet50.yaml --image=/root/semseg/dataset/cityscapes/leftImg8bit/val/frankfurt/frankfurt_000000_000294_leftImg8bit.png
cp -r ./figure/demo /home/cxiao/chaowei/ngc/workspace/jiachen_results/