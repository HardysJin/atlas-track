#!/bin/bash
dlav0_model='https://docs.google.com/uc?export=download&id=1SzsfCCR8DxlkNxX1DT4dTjvYsLrpdV8f'
dlav0_model_name="dlav0"
version=$1
data_source="../data/"
verify_source="../data/"
project_name="atlas-track"
script_path="$( cd "$(dirname $BASH_SOURCE)" ; pwd -P)"
project_path=${script_path}/..

declare -i success=0
declare -i inferenceError=1
declare -i verifyResError=2


function setAtcEnv() {

    if [[ ${version} = "c73" ]] || [[ ${version} = "C73" ]];then
        export install_path=/home/HwHiAiUser/Ascend/ascend-toolkit/latest
        export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
        export PYTHONPATH=${install_path}/atc/python/site-packages/te:${install_path}/atc/python/site-packages/topi:$PYTHONPATH
        export ASCEND_OPP_PATH=${install_path}/opp
        export LD_LIBRARY_PATH=${install_path}/atc/lib64:${LD_LIBRARY_PATH}
    elif [[ ${version} = "c75" ]] || [[ ${version} = "C75" ]];then
        export install_path=$HOME/Ascend/ascend-toolkit/latest
        export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
        export ASCEND_OPP_PATH=${install_path}/opp
        export PYTHONPATH=${install_path}/atc/python/site-packages:${install_path}/atc/python/site-packages/auto_tune.egg/auto_tune:${install_path}/atc/python/site-packages/schedule_search.egg:$PYTHONPATH
        export LD_LIBRARY_PATH=${install_path}/atc/lib64:${LD_LIBRARY_PATH}
    fi

    return 0
}

function setRunEnv() {

    if [[ ${version} = "c73" ]] || [[ ${version} = "C73" ]];then
        export LD_LIBRARY_PATH=
        export LD_LIBRARY_PATH=/home/HwHiAiUser/Ascend/acllib/lib64:/home/HwHiAiUser/ascend_ddk/arm/lib:${LD_LIBRARY_PATH}
        export PYTHONPATH=/home/HwHiAiUser/Ascend/ascend-toolkit/latest/arm64-linux_gcc7.3.0/pyACL/python/site-packages/acl:${PYTHONPATH}
    elif [[ ${version} = "c75" ]] || [[ ${version} = "C75" ]];then
        export LD_LIBRARY_PATH=
        export LD_LIBRARY_PATH=/home/HwHiAiUser/Ascend/acllib/lib64:/home/HwHiAiUser/ascend_ddk/arm/lib:${LD_LIBRARY_PATH}
        export PYTHONPATH=/home/HwHiAiUser/Ascend/ascend-toolkit/latest/arm64-linux/pyACL/python/site-packages/acl:${PYTHONPATH}
    fi

    return 0
}

function downloadOriginalModel() {
    mkdir -p ${project_path}/model/
    wget --no-check-certificate ${dlav0_model} -O ${project_path}/model/${dlav0_model_name}.om

    if [ $? -ne 0];then
        echo "Install dlav0.om failed, please check Network"
        return 1
    fi

    return 0
}

function main() {

    if [[ ${version}"x" = "x" ]];then
        echo "ERROR: version is invalid"
        return ${inferenceError}
    fi

    mkdir -p ${HOME}/models/${project_name}     
    # dlav0 conversion (-if find returns an empty string, then download model)
    if [[ $(find ${HOME}/models/${project_name} -name ${dlav0_model_name}".om")"x" = "x" ]];then 

        downloadOriginalModel
        if [ $? -ne 0 ];then
            echo "ERROR: download original dlav0 model failed"
            return ${inferenceError}
        fi

        # setAtcEnv
        export LD_LIBRARY_PATH=${install_path}/atc/lib64:$LD_LIBRARY_PATH
        if [ $? -ne 0 ];then
            echo "ERROR: set atc environment failed"
            return ${inferenceError}
        fi

        # Is this step necessary if we downloaded a .om?
        cd ${project_path}/model/
        # atc --framework=3 --model=${project_path}/model/yolo_model.pb --input_shape="input_1:1,416,416,3" --input_format=NHWC --output=${HOME}/models/${project_name}/${yolo_model_name} --output_type=FP32 --soc_version=Ascend310
        if [ $? -ne 0 ];then
            echo "ERROR: convert dlav0 model failed"
            return ${inferenceError}
        fi

        ln -sf ${HOME}/models/${project_name}/${dlav0_model_name}".om" ${project_path}/model/${dlav0_model_name}".om"
        if [ $? -ne 0 ];then
            echo "ERROR: failed to set dlav0 model soft connection"
            return ${inferenceError}
        fi
    else 
        ln -sf ${HOME}/models/${project_name}/${dlav0_model_name}".om" ${project_path}/model/${dlav0_model_name}".om"
        if [ $? -ne 0 ];then
            echo "ERROR: failed to set model soft connection"
            return ${inferenceError}
        fi
    fi
    cd ${project_path}


    # setRunEnv
    source ~/.bashrc
    if [ $? -ne 0 ];then
        echo "ERROR: set executable program running environment failed"
        return ${inferenceError}
    fi

    #*** change paths to images ***
    original_img=${project_path}/data/test.jpg
    verify_img=${project_path}/data/test_groundtruth.jpg
    mkdir -p ${project_path}/src/output
    rm ${project_path}/src/output/*
    python3 ${project_path}/src/main.py --input_image ${original_img}
    if [ $? -ne 0 ];then
        echo "ERROR: run failed. please check your project"
        return ${inferenceError}
    fi   
    out_img=${project_path}/src/output/test_output.jpg
    python3 ${script_path}/verify_result.py ${verify_img} ${out_img}
    if [ $? -ne 0 ];then
        echo "ERROR: The result of test 1 is wrong!"
        return ${verifyResError}
    fi   

    echo "********run test success********"

    return ${success}
}
main

#1. get/change path to test images 
#2. check if we can just download an .om model without using atc to convert
#3. test the script