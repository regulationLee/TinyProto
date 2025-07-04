#nohup python -u main.py -did 0 -data Cifar10 -m Ht0 -algo ePBFL -gr 100 -lam 10 -ssc -ppa -csf 0.00015 -go add_cifar10_epfl_adap_ssc_test > add_cifar10_epfl_adap_ssc_test.out 2>&1 &
#nohup python -u main.py -did 0 -data Cifar100 -m Ht0 -algo ePBFL -gr 100 -lam 10 -ssc -csf 0.0015 -go add_cifar100_epfl_adap_ssc_wo_ppa_test > add_cifar100_epfl_adap_ssc_wo_ppa_test.out 2>&1 &
#nohup python -u main.py -did 0 -data TinyImagenet -m Ht0 -algo ePBFL -gr 100 -lam 10 -ssc -ppa -csf 0.0015 -go add_TinyImagenet_epfl_adap_ssc_test > add_TinyImagenet_epfl_adap_ssc_test.out 2>&1 &
#
#nohup python -u main.py -did 0 -data Cifar10 -m Ht0 -algo FedTGP -gr 100 -lam 10 -ssc -ani -ppa -csf 0.001 -go add_cifar10_TGP_adap_ssc_ni_test > add_cifar10_TGP_adap_ssc_ni_test.out 2>&1 &
#nohup python -u main.py -did 1 -data Cifar100 -m Ht0 -algo FedTGP -gr 200 -lam 10 -ssc -ani -ppa -csf 0.0015 -go add_cifar100_TGP_adap_ssc_ni_test_01 > add_cifar100_TGP_adap_ssc_ni_test_01.out 2>&1 &
#nohup python -u main.py -did 0 -data Cifar100 -m Ht0 -algo FedTGP -gr 100 -lam 10 -ssc -csf 0.0015 -go add_cifar100_TGP_adap_ssc_wo_ppa_test > add_cifar100_TGP_adap_ssc_wo_ppa_test.out 2>&1 &
#nohup python -u main.py -did 0 -data TinyImagenet -m Ht0 -algo FedTGP -gr 100 -lam 10 -ssc -ani -ppa -csf 0.01 -go add_TinyImagenet_TGP_adap_ssc_ni_test > add_TinyImagenet_TGP_adap_ssc_ni_test.out 2>&1 &

#nohup python -u main.py -did 0 -data Cifar100 -m Ht0 -algo ePBFL -gr 100 -lam 10 -ssc -ppa -csf 0.001 -go add_cifar100_01test > add_cifar100_01test.out 2>&1 &
#nohup python -u main.py -did 1 -data TinyImagenet -m Ht0 -algo ePBFL -gr 100 -lam 10 -ssc -ppa -csf 0.001 -go add_TinyImagenet_01_test > add_TinyImagenet_01_test.out 2>&1 &
#nohup python -u main.py -did 0 -data Cifar10 -m Ht0 -algo ePBFL -gr 100 -lam 10 -ssc -ppa -csf 0.003 -go test > cifar10_test.out 2>&1 &

#!/bin/bash
## default hyperparameter
# lam: Cifar10-ePBFL 10 / Cifar10-TGP 10 / Cifar100-ePBFL 20 / Cifar100-TGP 20 / TinyImagenet-Proto 20  / TinyImagenet-TGP 20
# csf: Cifar10-ePBFL 0.00015 / Cifar10-TGP 0.001 / Cifar100-ePBFL 0.0015 / Cifar100-TGP 0.01 / TinyImagenet-Proto 0.0015  / TinyImagenet-TGP 0.01

exp=Proto_TGP
goal="final_result:paper"

data=Cifar100
result_dir="results/${data}_dir_${exp}_${goal}_results_$(date +%Y%m%d_%H%M)"
mkdir -p "$result_dir"

#data2=TinyImagenet
#result_dir_2="results/${data2}_dir_${exp}_${goal}_results_$(date +%Y%m%d_%H%M)"
#mkdir -p "$result_dir_2"

echo "result_dir created."

result_file="[results].csv"

if [ ! -f "$result_file" ]; then
  echo "exp_name,accuracy,min_dist,max_dist,avg_dist,min_ang,max_ang,avg_ang,avg_time" > "$result_file"
  echo "$result_file created."
else
  echo "$result_file existed."
fi

wait

### experiment list start #####

### Table 1. column G
## TinyImagenet
#algo=ePBFL
#lam=10
#csf=0.00015
#rcsr_list=(10 30 50 70 90)
#
#EXP_NAME="${data}_dir_Ht0_${algo}_lam_${lam}_ssc_csf_${csf}_01"
#nohup python -u main.py -did 0 -data ${data} -m Ht0 -algo ${algo} -lam ${lam} -ssc -ppa -csf ${csf} -go "$EXP_NAME" > "${EXP_NAME}.out" 2>&1 &
#
#for rcsr in "${rcsr_list[@]}"
#do
#  EXP_NAME="${data}_dir_Ht0_${algo}_lam_${lam}_ssc_csf_${csf}_csr_${rcsr}_01"
#  nohup python -u main.py -did 0 -data ${data} -m Ht0 -algo ${algo} -lam ${lam} -ssc -ppa -csf ${csf} -csr -rcsr ${rcsr} -go "$EXP_NAME" > "${EXP_NAME}.out" 2>&1 &
#done
#
#EXP_NAME="${data}_dir_Ht0_${algo}_lam_${lam}_ssc_csf_${csf}_02"
#nohup python -u main.py -did 0 -data ${data} -m Ht0 -algo ${algo} -lam ${lam} -ssc -ppa -csf ${csf} -go "$EXP_NAME" > "${EXP_NAME}.out" 2>&1 &
#
#for rcsr in "${rcsr_list[@]}"
#do
#  EXP_NAME="${data}_dir_Ht0_${algo}_lam_${lam}_ssc_csf_${csf}_csr_${rcsr}_02"
#  nohup python -u main.py -did 0 -data ${data} -m Ht0 -algo ${algo} -lam ${lam} -ssc -ppa -csf ${csf} -csr -rcsr ${rcsr} -go "$EXP_NAME" > "${EXP_NAME}.out" 2>&1 &
#done
#
#algo=FedTGP
#lam=10
#csf=0.001
#rcsr_list=(10 30 50 70 90)
#
#EXP_NAME="${data}_dir_Ht0_${algo}_lam_${lam}_ssc_csf_${csf}_01"
#nohup python -u main.py -did 1 -data ${data} -m Ht0 -algo ${algo} -lam ${lam} -ssc -ppa -csf ${csf} -go "$EXP_NAME" > "${EXP_NAME}.out" 2>&1 &
#
#for rcsr in "${rcsr_list[@]}"
#do
#  EXP_NAME="${data}_dir_Ht0_${algo}_lam_${lam}_ssc_csf_${csf}_csr_${rcsr}_01"
#  nohup python -u main.py -did 1 -data ${data} -m Ht0 -algo ${algo} -lam ${lam} -ssc -ppa -csf ${csf} -csr -rcsr ${rcsr} -go "$EXP_NAME" > "${EXP_NAME}.out" 2>&1 &
#done
#
#EXP_NAME="${data}_dir_Ht0_${algo}_lam_${lam}_ssc_csf_${csf}_02"
#nohup python -u main.py -did 1 -data ${data} -m Ht0 -algo ${algo} -lam ${lam} -ssc -ppa -csf ${csf} -go "$EXP_NAME" > "${EXP_NAME}.out" 2>&1 &
#
#for rcsr in "${rcsr_list[@]}"
#do
#  EXP_NAME="${data}_dir_Ht0_${algo}_lam_${lam}_ssc_csf_${csf}_csr_${rcsr}_02"
#  nohup python -u main.py -did 1 -data ${data} -m Ht0 -algo ${algo} -lam ${lam} -ssc -ppa -csf ${csf} -csr -rcsr ${rcsr} -go "$EXP_NAME" > "${EXP_NAME}.out" 2>&1 &
#done


#wait


### Table 1. column M
## TinyImagenet
#algo=ePBFL
#lam=10
#rcsr_list=(30 70 90)
#
#for rcsr in "${rcsr_list[@]}"
#do
#  EXP_NAME="${data2}_dir_Ht0_${algo}_lam_${lam}_csr_${rcsr}_01"
#  nohup python -u main.py -did 0 -data ${data2} -m Ht0 -algo ${algo} -lam ${lam} -csr -rcsr ${rcsr} -go "$EXP_NAME" > "${EXP_NAME}.out" 2>&1 &
#done
#
#for rcsr in "${rcsr_list[@]}"
#do
#  EXP_NAME="${data2}_dir_Ht0_${algo}_lam_${lam}_csr_${rcsr}_02"
#  nohup python -u main.py -did 0 -data ${data2} -m Ht0 -algo ${algo} -lam ${lam} -csr -rcsr ${rcsr} -go "$EXP_NAME" > "${EXP_NAME}.out" 2>&1 &
#done
#
#algo=FedTGP
#lam=10
#rcsr_list=(10 30 50 70 90)
#
#for rcsr in "${rcsr_list[@]}"
#do
#  EXP_NAME="${data2}_dir_Ht0_${algo}_lam_${lam}_csr_${rcsr}_01"
#  nohup python -u main.py -did 1 -data ${data2} -m Ht0 -algo ${algo} -lam ${lam} -csr -rcsr ${rcsr} -go "$EXP_NAME" > "${EXP_NAME}.out" 2>&1 &
#done
#
#for rcsr in "${rcsr_list[@]}"
#do
#  EXP_NAME="${data2}_dir_Ht0_${algo}_lam_${lam}_csr_${rcsr}_02"
#  nohup python -u main.py -did 1 -data ${data2} -m Ht0 -algo ${algo} -lam ${lam} -csr -rcsr ${rcsr} -go "$EXP_NAME" > "${EXP_NAME}.out" 2>&1 &
#done


#wait


### Table 1. column E
## Cifar 100
#algo=ePBFL
#lam=10
#csf=0.0015
#rcsr_list=(10 30 50 70 90)
#
#EXP_NAME="add_${data}_dir_Ht0_${algo}_lam_${lam}_test"
#nohup python -u main.py -did 0 -data ${data} -m Ht0 -algo ${algo} -lam ${lam} -go "$EXP_NAME" > "${EXP_NAME}.out" 2>&1 &
#
#EXP_NAME="add_${data}_dir_Ht0_${algo}_lam_${lam}_ssc_csf_${csf}_test"
#nohup python -u main.py -did 1 -data ${data} -m Ht0 -algo ${algo} -lam ${lam} -ssc -ppa -csf ${csf} -go "$EXP_NAME" > "${EXP_NAME}.out" 2>&1 &
#
#for rcsr in "${rcsr_list[@]}"
#do
#  EXP_NAME="${data}_dir_Ht0_${algo}_lam_${lam}_ssc_csf_${csf}_csr_${rcsr}_01"
#  nohup python -u main.py -did 0 -data ${data} -m Ht0 -algo ${algo} -lam ${lam} -ssc -ppa -csf ${csf} -csr -rcsr ${rcsr} -go "$EXP_NAME" > "${EXP_NAME}.out" 2>&1 &
#done
#
#EXP_NAME="${data}_dir_Ht0_${algo}_lam_${lam}_ssc_csf_${csf}_02"
#nohup python -u main.py -did 0 -data ${data} -m Ht0 -algo ${algo} -lam ${lam} -ssc -ppa -csf ${csf} -go "$EXP_NAME" > "${EXP_NAME}.out" 2>&1 &
#
#for rcsr in "${rcsr_list[@]}"
#do
#  EXP_NAME="${data}_dir_Ht0_${algo}_lam_${lam}_ssc_csf_${csf}_csr_${rcsr}_02"
#  nohup python -u main.py -did 0 -data ${data} -m Ht0 -algo ${algo} -lam ${lam} -ssc -ppa -csf ${csf} -csr -rcsr ${rcsr} -go "$EXP_NAME" > "${EXP_NAME}.out" 2>&1 &
#done
#
#algo=FedTGP
#lam=10
#csf=0.01
#rcsr_list=(10 30)
#
#EXP_NAME="add_${data}_dir_Ht0_${algo}_lam_${lam}_test"
#nohup python -u main.py -did 0 -data ${data} -m Ht0 -algo ${algo} -lam ${lam} -go "$EXP_NAME" > "${EXP_NAME}.out" 2>&1 &
#
#EXP_NAME="add_${data}_dir_Ht0_${algo}_lam_${lam}_ssc_csf_${csf}_test"
#nohup python -u main.py -did 0 -data ${data} -m Ht0 -algo ${algo} -lam ${lam} -ssc -ppa -csf ${csf} -go "$EXP_NAME" > "${EXP_NAME}.out" 2>&1 &
#
#for rcsr in "${rcsr_list[@]}"
#do
#  EXP_NAME="${data}_dir_Ht0_${algo}_lam_${lam}_ssc_csf_${csf}_csr_${rcsr}_00"
#  nohup python -u main.py -did 0 -data ${data} -m Ht0 -algo ${algo} -lam ${lam} -ssc -ppa -csf ${csf} -csr -rcsr ${rcsr} -go "$EXP_NAME" > "${EXP_NAME}.out" 2>&1 &
#done
#
#rcsr_list=(50 70 90)
#for rcsr in "${rcsr_list[@]}"
#do
#  EXP_NAME="${data}_dir_Ht0_${algo}_lam_${lam}_ssc_csf_${csf}_csr_${rcsr}_00"
#  nohup python -u main.py -did 1 -data ${data} -m Ht0 -algo ${algo} -lam ${lam} -ssc -ppa -csf ${csf} -csr -rcsr ${rcsr} -go "$EXP_NAME" > "${EXP_NAME}.out" 2>&1 &
#done

#EXP_NAME="${data}_dir_Ht0_${algo}_lam_${lam}_ssc_csf_${csf}_01"
#nohup python -u main.py -did 1 -data ${data} -m Ht0 -algo ${algo} -lam ${lam} -ssc -ppa -csf ${csf} -go "$EXP_NAME" > "${EXP_NAME}.out" 2>&1 &
#
#for rcsr in "${rcsr_list[@]}"
#do
#  EXP_NAME="${data}_dir_Ht0_${algo}_lam_${lam}_ssc_csf_${csf}_csr_${rcsr}_01"
#  nohup python -u main.py -did 1 -data ${data} -m Ht0 -algo ${algo} -lam ${lam} -ssc -ppa -csf ${csf} -csr -rcsr ${rcsr} -go "$EXP_NAME" > "${EXP_NAME}.out" 2>&1 &
#done
#
#EXP_NAME="${data}_dir_Ht0_${algo}_lam_${lam}_ssc_csf_${csf}_02"
#nohup python -u main.py -did 1 -data ${data} -m Ht0 -algo ${algo} -lam ${lam} -ssc -ppa -csf ${csf} -go "$EXP_NAME" > "${EXP_NAME}.out" 2>&1 &
#
#for rcsr in "${rcsr_list[@]}"
#do
#  EXP_NAME="${data}_dir_Ht0_${algo}_lam_${lam}_ssc_csf_${csf}_csr_${rcsr}_02"
#  nohup python -u main.py -did 1 -data ${data} -m Ht0 -algo ${algo} -lam ${lam} -ssc -ppa -csf ${csf} -csr -rcsr ${rcsr} -go "$EXP_NAME" > "${EXP_NAME}.out" 2>&1 &
#done
#
#
#
#wait



## Table 1. column K
#Cifar 100
#algo=ePBFL
#lam=10
#rcsr_list=(10 30 50 70 90)
#
#for rcsr in "${rcsr_list[@]}"
#do
#  EXP_NAME="${data}_dir_Ht0_${algo}_lam_${lam}_csr_${rcsr}_01"
#  nohup python -u main.py -did 0 -data ${data} -m Ht0 -algo ${algo} -gr 300 -lam ${lam} -csr -rcsr ${rcsr} -go "$EXP_NAME" > "${EXP_NAME}.out" 2>&1 &
#done
#
#for rcsr in "${rcsr_list[@]}"
#do
#  EXP_NAME="${data}_dir_Ht0_${algo}_lam_${lam}_csr_${rcsr}_02"
#  nohup python -u main.py -did 0 -data ${data} -m Ht0 -algo ${algo} -gr 300 -lam ${lam} -csr -rcsr ${rcsr} -go "$EXP_NAME" > "${EXP_NAME}.out" 2>&1 &
#done

#algo=FedTGP
#lam=10
#rcsr_list=(10 30 50 70 90)
#
#for rcsr in "${rcsr_list[@]}"
#do
#  EXP_NAME="${data}_dir_Ht0_${algo}_lam_${lam}_csr_${rcsr}_01"
#  nohup python -u main.py -did 0 -data ${data} -m Ht0 -algo ${algo} -lam ${lam} -csr -rcsr ${rcsr} -go "$EXP_NAME" > "${EXP_NAME}.out" 2>&1 &
#done
#
#for rcsr in "${rcsr_list[@]}"
#do
#  EXP_NAME="${data}_dir_Ht0_${algo}_lam_${lam}_csr_${rcsr}_02"
#  nohup python -u main.py -did 1 -data ${data} -m Ht0 -algo ${algo} -lam ${lam} -csr -rcsr ${rcsr} -go "$EXP_NAME" > "${EXP_NAME}.out" 2>&1 &
#done



#wait
#
#
### Table 5
## Cifar 100
#fd_list=(100 200 400 600 800 1000)
#rcsr_list=(10 50)
#lam=10
#
#algo=ePBFL
#csf=0.0015
#for fd in "${fd_list[@]}"
#do
#  for rcsr in "${rcsr_list[@]}"
#  do
#    EXP_NAME="${data}_dir_Ht0_${algo}_fd_${fd}_lam_${lam}_ssc_csf_${csf}_csr_${rcsr}_00"
#    nohup python -u main.py -did 0 -data ${data} -m Ht0 -algo ${algo} -lam ${lam} -fd ${fd} -ssc -ppa -csf ${csf} -csr -rcsr ${rcsr} -go "$EXP_NAME" > "${EXP_NAME}.out" 2>&1 &
#  done
#done
#
#algo=FedTGP
#csf=0.01
#for fd in "${fd_list[@]}"
#do
#  for rcsr in "${rcsr_list[@]}"
#  do
#    EXP_NAME="${data}_dir_Ht0_${algo}_fd_${fd}_lam_${lam}_ssc_csf_${csf}_csr_${rcsr}_00"
#    nohup python -u main.py -did 1 -data ${data} -m Ht0 -algo ${algo} -lam ${lam} -fd ${fd} -ssc -ppa -csf ${csf} -csr -rcsr ${rcsr} -go "$EXP_NAME" > "${EXP_NAME}.out" 2>&1 &
#  done
#done


#wait
#
#
### Table 5
## Cifar 100
#fd_list=(100 200 400 600 800 1000)
#rcsr_list=(10 50)
#lam=10
#
#algo=ePBFL
#csf=0.0015
#for fd in "${fd_list[@]}"
#do
#  for rcsr in "${rcsr_list[@]}"
#  do
#    EXP_NAME="${data}_dir_Ht0_${algo}_fd_${fd}_lam_${lam}_ssc_csf_${csf}_csr_${rcsr}_01"
#    nohup python -u main.py -did 0 -data ${data} -m Ht0 -algo ${algo} -lam ${lam} -fd ${fd} -ssc -ppa -csf ${csf} -csr -rcsr ${rcsr} -go "$EXP_NAME" > "${EXP_NAME}.out" 2>&1 &
#  done
#done
#
#algo=FedTGP
#csf=0.01
#for fd in "${fd_list[@]}"
#do
#  for rcsr in "${rcsr_list[@]}"
#  do
#    EXP_NAME="${data}_dir_Ht0_${algo}_fd_${fd}_lam_${lam}_ssc_csf_${csf}_csr_${rcsr}_01"
#    nohup python -u main.py -did 1 -data ${data} -m Ht0 -algo ${algo} -lam ${lam} -fd ${fd} -ssc -ppa -csf ${csf} -csr -rcsr ${rcsr} -go "$EXP_NAME" > "${EXP_NAME}.out" 2>&1 &
#  done
#done
#
#
#wait
#
#
### Table 5
## Cifar 100
#fd_list=(100 200 400 600 800 1000)
#rcsr_list=(10 50)
#lam=10
#
#algo=ePBFL
#csf=0.0015
#for fd in "${fd_list[@]}"
#do
#  for rcsr in "${rcsr_list[@]}"
#  do
#    EXP_NAME="${data}_dir_Ht0_${algo}_fd_${fd}_lam_${lam}_ssc_csf_${csf}_csr_${rcsr}_02"
#    nohup python -u main.py -did 0 -data ${data} -m Ht0 -algo ${algo} -lam ${lam} -fd ${fd} -ssc -ppa -csf ${csf} -csr -rcsr ${rcsr} -go "$EXP_NAME" > "${EXP_NAME}.out" 2>&1 &
#  done
#done
#
#algo=FedTGP
#csf=0.01
#for fd in "${fd_list[@]}"
#do
#  for rcsr in "${rcsr_list[@]}"
#  do
#    EXP_NAME="${data}_dir_Ht0_${algo}_fd_${fd}_lam_${lam}_ssc_csf_${csf}_csr_${rcsr}_02"
#    nohup python -u main.py -did 1 -data ${data} -m Ht0 -algo ${algo} -lam ${lam} -fd ${fd} -ssc -ppa -csf ${csf} -csr -rcsr ${rcsr} -go "$EXP_NAME" > "${EXP_NAME}.out" 2>&1 &
#  done
#done

## Table 3
# Cifar 100

#alpha_list=(0.01 0.5 1.0)
#alpha_list=(0.5)
#trial_list=(00 01 02)
#lam=10
#rcsr=10
#
#for alpha in "${alpha_list[@]}"
#do
#  cd ..
#  cd dataset
#  python generate_cifar100.py noniid - dir 20 ${alpha}
#  wait
#  cd ..
#  cd _TinyProto
#
#  wait
#
#  algo=ePBFL
#  csf=0.0015
#  for trial in "${trial_list[@]}"
#  do
#    EXP_NAME="${data}_dir_alpha_${alpha}_Ht0_${algo}_lam_${lam}_ssc_csf_${csf}_csr_${rcsr}_${trial}"
#    nohup python -u main.py -did 0 -data ${data} -m Ht0 -algo ${algo} -lam ${lam} -ssc -ppa -csf ${csf} -csr -rcsr ${rcsr} -go "$EXP_NAME" > "${EXP_NAME}.out" 2>&1 &
#
#    EXP_NAME="${data}_dir_alpha_${alpha}_Ht0_${algo}_${trial}"
#    nohup python -u main.py -did 1 -data ${data} -m Ht0 -algo ${algo} -go "$EXP_NAME" > "${EXP_NAME}.out" 2>&1 &
#  done
#
#  algo=FedTGP
#  csf=0.01
#  for trial in "${trial_list[@]}"
#  do
#    EXP_NAME="${data}_dir_alpha_${alpha}_Ht0_${algo}_lam_${lam}_ssc_csf_${csf}_csr_${rcsr}_${trial}"
#    nohup python -u main.py -did 1 -data ${data} -m Ht0 -algo ${algo} -lam ${lam} -ssc -ppa -csf ${csf} -csr -rcsr ${rcsr} -go "$EXP_NAME" > "${EXP_NAME}.out" 2>&1 &
#
#    EXP_NAME="${data}_dir_alpha_${alpha}_Ht0_${algo}_${trial}"
#    nohup python -u main.py -did 0 -data ${data} -m Ht0 -algo ${algo} -go "$EXP_NAME" > "${EXP_NAME}.out" 2>&1 &
#  done

#  for trial in "${trial_list[@]}"
#  do
#    nohup python -u main.py -did 0 -data Cifar100 -m Ht0 -algo LG-FedAvg -gr 300 -go Cifar100_dir_alpha_${alpha}_Ht0_LGFedAvg_${trial} > Cifar100_dir_alpha_${alpha}_Ht0_LGFedAvg_${trial}.out 2>&1 &
#    nohup python -u main.py -did 1 -data Cifar100 -m Ht0 -algo FML -gr 300 -go Cifar100_dir_alpha_${alpha}_Ht0_FML_${trial} > Cifar100_dir_alpha_${alpha}_Ht0_FML_${trial}.out 2>&1 &
#
#    nohup python -u main.py -did 0 -data Cifar100 -m Ht0 -algo FedKD -gr 300 -go Cifar100_dir_alpha_${alpha}_Ht0_FedKD_${trial} > Cifar100_dir_alpha_${alpha}_Ht0_FedKD_${trial}.out 2>&1 &
#    nohup python -u main.py -did 1 -data Cifar100 -m Ht0 -algo FedDistill -gr 300 -go Cifar100_dir_alpha_${alpha}_Ht0_FedDistill_${trial} > Cifar100_dir_alpha_${alpha}_Ht0_FedDistill_${trial}.out 2>&1 &
#  done

#  wait

#done

cd ..
cd dataset
python generate_cifar100.py noniid - dir 20 0.1
wait
cd ..
cd _TinyProto

wait

## Table 5
# Cifar 100
fd_list=(64 1024)
trial_list=(00 01 02)
for fd in "${fd_list[@]}"
do
  for trial in "${trial_list[@]}"
  do
    nohup python -u main.py -did 0 -data Cifar100 -m Ht0 -algo LG-FedAvg -fd ${fd} -go Cifar100_dir_Ht0_LGFedAvg_fd_${fd}_${trial} > Cifar100_dir_Ht0_LGFedAvg_fd_${fd}_${trial}.out 2>&1 &
    nohup python -u main.py -did 1 -data Cifar100 -m Ht0 -algo FML -fd ${fd} -go Cifar100_dir_Ht0_FML_fd_${fd}_${trial} > Cifar100_dir_Ht0_FML_fd_${fd}_${trial}.out 2>&1 &
    nohup python -u main.py -did 1 -data Cifar100 -m Ht0 -algo FedKD -fd ${fd} -go Cifar100_dir_Ht0_FedKD_fd_${fd}_${trial} > Cifar100_dir_Ht0_FedKD_fd_${fd}_${trial}.out 2>&1 &
    nohup python -u main.py -did 0 -data Cifar100 -m Ht0 -algo FedDistill -fd ${fd} -go Cifar100_dir_Ht0_FedDistill_fd_${fd}_${trial} > Cifar100_dir_Ht0_FedDistill_fd_${fd}_${trial}.out 2>&1 &
    nohup python -u main.py -did 0 -data Cifar100 -m Ht0 -algo ePBFL -fd ${fd} -go Cifar100_dir_Ht0_ePBFL_fd_${fd}_${trial} > Cifar100_dir_Ht0_ePBFL_fd_${fd}_${trial}.out 2>&1 &
    nohup python -u main.py -did 1 -data Cifar100 -m Ht0 -algo FedTGP -fd ${fd} -go Cifar100_dir_Ht0_FedTGP_fd_${fd}_${trial} > Cifar100_dir_Ht0_FedTGP_fd_${fd}_${trial}.out 2>&1 &
  done
  wait
done
#
#wait
#
#nc_list=(50 100)
#lam=10
#rcsr=50
#
#for nc in "${nc_list[@]}"
#do
#  cd ..
#  cd dataset
#  python generate_cifar100.py noniid - dir ${nc} 0.1
#  wait
#  cd ..
#  cd _TinyProto
#
#  wait
#
#  for trial in "${trial_list[@]}"
#  do
#    ## Table 1. column C
#    algo=ePBFL
#    csf=0.0015
#    EXP_NAME="${data}_dir_Ht0_${algo}_lam_${lam}_nc_${nc}_ssc_csf_${csf}_csr_${rcsr}_${trial}"
#    nohup python -u main.py -did 0 -data ${data} -m Ht0 -algo ${algo} -gr 500 -nc ${nc} -lam ${lam} -ssc -ppa -csf ${csf} -csr -rcsr ${rcsr} -go "$EXP_NAME" > "${EXP_NAME}.out" 2>&1 &
#
#    algo=FedTGP
#    csf=0.01
#    EXP_NAME="${data}_dir_Ht0_${algo}_lam_${lam}_nc_${nc}_ssc_csf_${csf}_csr_${rcsr}_${trial}"
#    nohup python -u main.py -did 1 -data ${data} -m Ht0 -algo ${algo} -gr 500 -nc ${nc} -lam ${lam} -ssc -ppa -csf ${csf} -csr -rcsr ${rcsr} -go "$EXP_NAME" > "${EXP_NAME}.out" 2>&1 &
#  done
#
#  wait
#
#done
#
#wait
#
#cd ..
#cd dataset
#python generate_cifar100.py noniid - dir 20 0.1
#wait
#cd ..
#cd _TinyProto
#
#rcsr_list=(10 30 50 70 90)
#for rcsr in "${rcsr_list[@]}"
#do
#  algo=ePBFL
#  csf=0.0015
#  EXP_NAME="${data}_dir_Ht0_${algo}_lam_${lam}_ssc_csf_${csf}_csr_${rcsr}_test"
#  nohup python -u main.py -did 0 -data ${data} -m Ht0 -algo ${algo} -lam ${lam} -ssc -ppa -csf ${csf} -csr -rcsr ${rcsr} -go "$EXP_NAME" > "${EXP_NAME}.out" 2>&1 &
#
#  algo=FedTGP
#  csf=0.01
#  EXP_NAME="${data}_dir_Ht0_${algo}_lam_${lam}_ssc_csf_${csf}_csr_${rcsr}_test"
#  nohup python -u main.py -did 1 -data ${data} -m Ht0 -algo ${algo} -lam ${lam} -ssc -ppa -csf ${csf} -csr -rcsr ${rcsr} -go "$EXP_NAME" > "${EXP_NAME}.out" 2>&1 &
#done
#
#rcsr_list=(10 30 50 70 90)
#for rcsr in "${rcsr_list[@]}"
#do
#  algo=ePBFL
#  EXP_NAME="${data}_dir_Ht0_${algo}_lam_${lam}_csr_${rcsr}_test"
#  nohup python -u main.py -did 1 -data ${data} -m Ht0 -algo ${algo} -lam ${lam} -csr -rcsr ${rcsr} -go "$EXP_NAME" > "${EXP_NAME}.out" 2>&1 &
#
#  algo=FedTGP
#  csf=0.01
#  EXP_NAME="${data}_dir_Ht0_${algo}_lam_${lam}_csr_${rcsr}_test"
#  nohup python -u main.py -did 0 -data ${data} -m Ht0 -algo ${algo} -lam ${lam} -csr -rcsr ${rcsr} -go "$EXP_NAME" > "${EXP_NAME}.out" 2>&1 &
#done

### experiment list end #####

wait

mv "${data}"* "$result_dir/"
#mv "${data2}"* "$result_dir_2/"
echo "The resulting log files has been moved."







########## verify original code
#### experiment list start #####
#lam=10
#data_1=Cifar100
#algo=ePBFL
#EXP_NAME="${data}_dir_Ht0_${algo}_lam_${lam}_test"
#nohup python -u main.py -did 0 -data ${data} -m Ht0 -algo ${algo} -gr 300 -lam $lam -go "$EXP_NAME" > "${EXP_NAME}.out" 2>&1 &
#
#EXP_NAME="${data}_dir_Ht0_${algo}_lam_${lam}_cpkd_beta_ij_test"
#nohup python -u main.py -did 0 -data ${data} -m Ht0 -algo ${algo} -gr 300 -lam $lam -cpkd -csw beta_ij -go "$EXP_NAME" > "${EXP_NAME}.out" 2>&1 &
#
#EXP_NAME="${data_1}_dir_Ht0_${algo}_lam_${lam}_test"
#nohup python -u main.py -did 0 -data ${data_1} -m Ht0 -algo ${algo} -gr 300 -lam $lam -go "$EXP_NAME" > "${EXP_NAME}.out" 2>&1 &
#
#EXP_NAME="${data_1}_dir_Ht0_${algo}_lam_${lam}_cpkd_beta_ij_test"
#nohup python -u main.py -did 0 -data ${data_1} -m Ht0 -algo ${algo} -gr 300 -lam $lam -cpkd -csw beta_ij -go "$EXP_NAME" > "${EXP_NAME}.out" 2>&1 &
#
#algo=FedTGP
#EXP_NAME="${data}_dir_Ht0_${algo}_lam_${lam}_csf_${csf}_test"
#nohup python -u main.py -did 1 -data ${data} -m Ht0 -algo ${algo} -gr 300 -lam $lam -go "$EXP_NAME" > "${EXP_NAME}.out" 2>&1 &
#
#EXP_NAME="${data}_dir_Ht0_${algo}_lam_${lam}_csf_${csf}_cpkd_beta_ij_test"
#nohup python -u main.py -did 1 -data ${data} -m Ht0 -algo ${algo} -gr 300 -lam $lam -cpkd -csw beta_ij -go "$EXP_NAME" > "${EXP_NAME}.out" 2>&1 &
#
#EXP_NAME="${data_1}_dir_Ht0_${algo}_lam_${lam}_csf_${csf}_test"
#nohup python -u main.py -did 1 -data ${data_1} -m Ht0 -algo ${algo} -gr 300 -lam $lam -go "$EXP_NAME" > "${EXP_NAME}.out" 2>&1 &
#
#EXP_NAME="${data_1}_dir_Ht0_${algo}_lam_${lam}_csf_${csf}_cpkd_beta_ij_test"
#nohup python -u main.py -did 1 -data ${data_1} -m Ht0 -algo ${algo} -gr 300 -lam $lam -cpkd -csw beta_ij -go "$EXP_NAME" > "${EXP_NAME}.out" 2>&1 &
#
###### experiment list end #####

#EXP_NAME="${data}_dir_Ht0_${algo}_lam_${lam}_csf_${csf}_cpkd_beta_ij_test"
#nohup python -u main.py -did 0 -data ${data} -m Ht0 -algo ${algo} -gr 300 -ppa -lam $lam -csf $csf -cpkd -csw beta_ij -go "$EXP_NAME" > "${EXP_NAME}.out" 2>&1 &
#
#EXP_NAME="${data}_dir_Ht0_${algo}_lam_${lam}_csf_${csf}_cpkd_beta_ij_test"
#nohup python -u main.py -did 1 -data ${data} -m Ht0 -algo ${algo} -gr 300 -lam $lam -csf $csf -cpkd -csw beta_ij -go "$EXP_NAME" > "${EXP_NAME}.out" 2>&1 &
#
#EXP_NAME="${data}_dir_Ht0_${algo}_lam_${lam}_csf_${csf}_cpkd_beta_ij_test"
#nohup python -u main.py -did 0 -data ${data} -m Ht0 -algo ${algo} -gr 300 -ppa -lam $lam -csf $csf -cpkd -csw beta_ij -go "$EXP_NAME" > "${EXP_NAME}.out" 2>&1 &
#
#EXP_NAME="${data}_dir_Ht0_${algo}_lam_${lam}_csf_${csf}_cpkd_beta_ij_test"
#nohup python -u main.py -did 1 -data ${data} -m Ht0 -algo ${algo} -gr 300 -ppa -lam $lam -csf $csf -cpkd -csw beta_ij -go "$EXP_NAME" > "${EXP_NAME}.out" 2>&1 &

