#!/bin/bash
# python train_distill.py --only_sample --resume_checkpoint ../../autodl-tmp/kd_distill/distill_t128_s64_260319_231716/checkpoint-5000
# python train_distill.py --only_sample --resume_checkpoint ../../autodl-tmp/kd_distill/distill_t128_s64_260319_231716/checkpoint-10000
# python train_distill.py --only_sample --resume_checkpoint ../../autodl-tmp/kd_distill/distill_t128_s64_260319_231716/checkpoint-15000
# python train_distill.py --only_sample --resume_checkpoint ../../autodl-tmp/kd_distill/distill_t128_s64_260319_231716/checkpoint-20000
# python train_distill.py --only_sample --resume_checkpoint ../../autodl-tmp/kd_distill/distill_t128_s64_260319_231716/checkpoint-25000
# python train_distill.py --only_sample --resume_checkpoint ../../autodl-tmp/kd_distill/distill_t256_s128_260320_100630/checkpoint-25000
# python train_distill.py --only_sample --resume_checkpoint ../../autodl-tmp/kd_distill/distill_t128_s64_260319_231716/checkpoint-35000
python train_distill.py
curl "https://sctapi.ftqq.com/SCT313662TGZ7JRPbisBQfDZbabO1Kmmdt.send?title=训练完成&desp=Python脚本已执行完毕channel=9"
sleep 3
/usr/bin/shutdown
sleep 3
/usr/bin/shutdown
sleep 3
/usr/bin/shutdown
sleep 3
/usr/bin/shutdown
sleep 3
/usr/bin/shutdown
sleep 3
/usr/bin/shutdown
sleep 3
/usr/bin/shutdown
sleep 3
curl "https://sctapi.ftqq.com/SCT313662TGZ7JRPbisBQfDZbabO1Kmmdt.send?title=服务器关闭失败&desp=服务器关闭失败channel=9"