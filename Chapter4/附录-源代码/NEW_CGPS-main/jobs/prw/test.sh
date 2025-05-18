#!/bin/bash
# 第一部分运行了一个分布式测试脚本，并保存了测试结果。
# 第二部分运行了一个 Python 脚本来处理这些测试结果，并将输出保存到日志文件中。
../../tools/dist_test.sh ../../configs/cgps/prw.py ../../work_dirs/latest.pth 1 --out results_1000.pkl >log_tmp.txt 2>&1 
python ../../tools/test_results_prw.py >result.txt 2>&1
