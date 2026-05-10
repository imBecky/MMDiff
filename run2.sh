#!/usr/bin/env bash
# python eval_ckps.py
python plot_eval_ckps.py --eval-dir ../../autodl-tmp/classifier/exp_0423-1312_exp_es_s84_mod_hsi_rgb_lidar/interested/eval_all_checkpoints
python eval_ckps.py --run-dir ../../autodl-tmp/classifier/exp_0423-1426_exp_es_s168_mod_hsi_rgb_lidar/interested
python eval_ckps.py --run-dir ../../autodl-tmp/classifier/exp_0423-1546_exp_es_s336_mod_hsi_rgb_lidar/interested
python eval_ckps.py --run-dir ../../autodl-tmp/classifier/exp_0423-1656_exp_es_s672_mod_hsi_rgb_lidar/interested
do_shutdown() {
  sleep 3
  local _i
  curl "https://sctapi.ftqq.com/SCT313662TGZ7JRPbisBQfDZbabO1Kmmdt.send?title=训练完成&desp=Python脚本已执行完毕channel=9"
  for _i in 1 2 3 4 5 6 7 8 9; do
    /usr/bin/shutdown
    sleep 3
  done
  curl -fsS "https://sctapi.ftqq.com/SCT313662TGZ7JRPbisBQfDZbabO1Kmmdt.send?title=服务器关闭失败&desp=服务器关闭失败channel=9"
}
do_shutdown