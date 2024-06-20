FAMILY_NAME="MPNN_SameAsGine" MUP_BASE_PATH=/net/group/all-ipu/blazej_ckpt/MPNN++_BackToOG_10M_20240115_144553/base_shape.yaml
    RUN_DIR=/net/group/all-ipu/blazej_ckpt/AdmetFiltered_MPNN++_BackToOG_10M_SameHidAsGINE_20240325_130123
    CONFIG_NAME=AdmetFiltered_MPNN_10M_LargeMix_SameHidAsGINE.yaml
        RUN_NAME_SUFFIX="${FAMILY_NAME}_50M"
        CKPT_NAME_FOR_TESTING=AdmetFiltered_MPNN++_BackToOG_10M_SameHidAsGINE_20240325_130123.ckpt
        

python graphium/cli/get_final_fingerprints.py \
    --config-path=$RUN_DIR \
    --config-name=$CONFIG_NAME \
    datamodule.args.featurization_backend=threading \
    datamodule.args.featurization.max_num_atoms=99999 \
    architecture.graph_output_nn.graph.last_normalization=layer_norm \
    accelerator.type=cpu \
    architecture.mup_load_or_save=load \
    architecture.mup_base_path=$MUP_BASE_PATH \
    trainer.model_checkpoint.dirpath=$RUN_DIR \
    +ckpt_name_for_testing=$CKPT_NAME_FOR_TESTING \
    +run_name_suffix=$RUN_NAME_SUFFIX

# FAMILY_NAME="GINE_AdmetFiltered_k10" MUP_BASE_PATH=/net/group/all-ipu/blazej_ckpt/base_shape_10M_GINE/base_shape.yaml
#     RUN_DIR=/net/group/all-ipu/blazej_ckpt/GINE_10M_k10_20240325_123752
#     CONFIG_NAME=AdmetFiltered_GINE_10M_LargeMix_hid128_k10.yaml
#         RUN_NAME_SUFFIX="${FAMILY_NAME}_10M"
#         CKPT_NAME_FOR_TESTING=GINE_10M_k10_20240325_123752.ckpt

# FAMILY_NAME="GINE_AdmetFiltered_k2" MUP_BASE_PATH=/net/group/all-ipu/blazej_ckpt/base_shape_10M_GINE/base_shape.yaml
#     RUN_DIR=/net/group/all-ipu/blazej_ckpt/GINE_10M_k2_20240325_123610
#     CONFIG_NAME=AdmetFiltered_GINE_10M_LargeMix_hid128_k2.yaml
#         RUN_NAME_SUFFIX="${FAMILY_NAME}_10M"
#         CKPT_NAME_FOR_TESTING=GINE_10M_k2_20240325_123610.ckpt

# FAMILY_NAME="GINE_AdmetFiltered_k1" MUP_BASE_PATH=/net/group/all-ipu/blazej_ckpt/base_shape_10M_GINE/base_shape.yaml
#     RUN_DIR=/net/group/all-ipu/blazej_ckpt/GINE_10M_k1_20240325_123427
#     CONFIG_NAME=AdmetFiltered_GINE_10M_LargeMix_hid128_k1.yaml
#         RUN_NAME_SUFFIX="${FAMILY_NAME}_10M"
#         CKPT_NAME_FOR_TESTING=GINE_10M_k1_20240325_123427.ckpt

# FAMILY_NAME="GINE_AdmetFiltered_seed11" MUP_BASE_PATH=/net/group/all-ipu/blazej_ckpt/base_shape_10M_GINE/base_shape.yaml
#     RUN_DIR=/net/group/all-ipu/blazej_ckpt/GINE_10M_Seed11_20240325_123214
#     CONFIG_NAME=AdmetFiltered_GINE_10M_LargeMix_hid128_SEED11.yaml
#         RUN_NAME_SUFFIX="${FAMILY_NAME}_10M"
#         CKPT_NAME_FOR_TESTING=GINE_10M_Seed11_20240325_123214.ckpt

# FAMILY_NAME="GINE_AdmetFiltered_seed6" MUP_BASE_PATH=/net/group/all-ipu/blazej_ckpt/base_shape_10M_GINE/base_shape.yaml
#     RUN_DIR=/net/group/all-ipu/blazej_ckpt/GINE_10M_Seed6_20240325_122510
#     CONFIG_NAME=AdmetFiltered_GINE_10M_LargeMix_hid128_SEED6.yaml
#         RUN_NAME_SUFFIX="${FAMILY_NAME}_10M"
#         CKPT_NAME_FOR_TESTING=GINE_10M_Seed6_20240325_122510.ckpt

# FAMILY_NAME="GINE_AdmetFiltered_maxpool" MUP_BASE_PATH=/net/group/all-ipu/blazej_ckpt/base_shape_10M_GINE/base_shape.yaml
#     RUN_DIR=/net/group/all-ipu/blazej_ckpt/GINE_10M_k5_MaxPool_20240325_122510
#     CONFIG_NAME=AdmetFiltered_GINE_10M_LargeMix_hid128_k5_MaxPool.yaml
#         RUN_NAME_SUFFIX="${FAMILY_NAME}_10M"
#         CKPT_NAME_FOR_TESTING=GINE_10M_k5_MaxPool_20240325_122510.ckpt

# FAMILY_NAME="GINE_AdmetFiltered_meanpool" MUP_BASE_PATH=/net/group/all-ipu/blazej_ckpt/base_shape_10M_GINE/base_shape.yaml
#     RUN_DIR=/net/group/all-ipu/blazej_ckpt/GINE_10M_k5_MeanPool_20240325_122510
#     CONFIG_NAME=AdmetFiltered_GINE_10M_LargeMix_hid128_k5_MeanPool.yaml
#         RUN_NAME_SUFFIX="${FAMILY_NAME}_10M"
#         CKPT_NAME_FOR_TESTING=GINE_10M_k5_MeanPool_20240325_122510.ckpt


# FAMILY_NAME="MPNN_AdmetFiltered_lastnorm" MUP_BASE_PATH="/net/group/all-ipu/blazej_ckpt/MPNN++_BackToOG_10M_20240115_144553/base_shape.yaml"
#     RUN_DIR=/net/group/all-ipu/blazej_ckpt/AdmetFiltered_MPNN++_BackToOG_10M_20240126_194220
#     CONFIG_NAME=AdmetFiltered_MPNN_10M_LargeMix_hid128.yaml
#         RUN_NAME_SUFFIX="${FAMILY_NAME}_10M"
#         CKPT_NAME_FOR_TESTING=AdmetFiltered_MPNN++_BackToOG_10M_20240126_194220.ckpt

# FAMILY_NAME="GINE_filtered_pcba" MUP_BASE_PATH=/net/group/all-ipu/blazej_ckpt/base_shape_10M_GINE_pcba/base_shape.yaml
    # RUN_DIR=/net/group/all-ipu/blazej_ckpt/GINE_40M_PCBAonly_20240306_143013
    # CONFIG_NAME=AdmetFiltered_GINE_40M_PCBAonly.yaml
    # RUN_NAME_SUFFIX="${FAMILY_NAME}_lr3e-4_40M"
    # CKPT_NAME_FOR_TESTING=GINE_40M_PCBAonly_20240306_143013.ckpt

    # RUN_DIR=/net/group/all-ipu/blazej_ckpt/GINE_40M_PCBAonly_20240306_171318
    # CONFIG_NAME=AdmetFiltered_GINE_40M_PCBAonly.yaml
    # RUN_NAME_SUFFIX="${FAMILY_NAME}_lr4e-4_40M"
    # CKPT_NAME_FOR_TESTING=GINE_40M_PCBAonly_20240306_171318.ckpt

# FAMILY_NAME="GINE_filtered_pcba" MUP_BASE_PATH=/net/group/all-ipu/blazej_ckpt/base_shape_10M_GINE_pcba/base_shape.yaml
#     RUN_DIR=/net/group/all-ipu/blazej_ckpt/GINE_10M_pcba_20240305_203810
#     CONFIG_NAME=AdmetFiltered_GINE_10M_PCBAonly.yaml
#         RUN_NAME_SUFFIX="${FAMILY_NAME}_10M"
#         CKPT_NAME_FOR_TESTING=GINE_10M_pcba_20240305_203810.ckpt

# FAMILY_NAME="GINE_AdmetFiltered" MUP_BASE_PATH=/net/group/all-ipu/blazej_ckpt/base_shape_10M_GINE/base_shape.yaml
    # RUN_DIR=/net/group/all-ipu/blazej_ckpt/GINE_10M_AdmetFiltered_20240214_125306
    # CONFIG_NAME=AdmetFiltered_GINE_10M_LargeMix_hid128.yaml
        # RUN_NAME_SUFFIX="${FAMILY_NAME}_10M"
        # CKPT_NAME_FOR_TESTING=GINE_10M_AdmetFiltered_20240214_125306.ckpt

# FAMILY_NAME="GCN_AdmetFiltered_lastnorm" MUP_BASE_PATH=/net/group/all-ipu/blazej_ckpt/base_shape_10M_GCN/base_shape.yaml
#     RUN_DIR=/net/group/all-ipu/blazej_ckpt/GCN_10M_AdmetFiltered_20240214_124029
#     CONFIG_NAME=AdmetFiltered_GCN_10M_LargeMix_hid128.yaml
#         RUN_NAME_SUFFIX="${FAMILY_NAME}_10M"
#         CKPT_NAME_FOR_TESTING=GCN_10M_AdmetFiltered_20240214_124029.ckpt 

# FAMILY_NAME="GINE" MUP_BASE_PATH="/net/group/all-ipu/blazej_ckpt/base_shape_10M_GINE/base_shape.yaml"
#     RUN_DIR=/net/group/all-ipu/blazej_ckpt/GINE_10M_20240126_194654
#     CONFIG_NAME=GINE_10M_LargeMix_hid128.yaml
#         RUN_NAME_SUFFIX="${FAMILY_NAME}_10M"
#         CKPT_NAME_FOR_TESTING=GINE_10M_20240126_194654.ckpt


# FAMILY_NAME="GCN" MUP_BASE_PATH="/net/group/all-ipu/blazej_ckpt/base_shape_10M_GCN/base_shape.yaml"
#     RUN_DIR=/net/group/all-ipu/blazej_ckpt/GCN_10M_20240126_194703
#     CONFIG_NAME=GCN_10M_LargeMix_hid128.yaml
#         RUN_NAME_SUFFIX="${FAMILY_NAME}_10M"
#         CKPT_NAME_FOR_TESTING=GCN_10M_20240126_194703.ckpt


# FAMILY_NAME="FilteredAdmet" MUP_BASE_PATH="/net/group/all-ipu/blazej_ckpt/MPNN++_BackToOG_10M_20240115_144553/base_shape.yaml"
#     RUN_DIR=/net/group/all-ipu/blazej_ckpt/AdmetFiltered_MPNN++_BackToOG_10M_20240126_194220
#     CONFIG_NAME=AdmetFiltered_MPNN_10M_LargeMix_hid128.yaml
#         RUN_NAME_SUFFIX="${FAMILY_NAME}_10M"
#         CKPT_NAME_FOR_TESTING=AdmetFiltered_MPNN++_BackToOG_10M_20240126_194220.ckpt


# FAMILY_NAME="BugFixOg_2e-1-g25-loss_g25do" MUP_BASE_PATH="/net/group/all-ipu/blazej_ckpt/MPNN++_BackToOG_10M_20240115_144553/base_shape.yaml"
#     RUN_DIR=/net/group/all-ipu/blazej_ckpt/MPNN++_BackToOG_35M_20240124_003925
#     CONFIG_NAME=MPNN_35M_LargeMix_hid256.yaml
        # RUN_NAME_SUFFIX="${FAMILY_NAME}_35M_best"
        # CKPT_NAME_FOR_TESTING=MPNN++_BackToOG_35M_20240124_003925.ckpt

        # RUN_NAME_SUFFIX="${FAMILY_NAME}_35M_last"
        # CKPT_NAME_FOR_TESTING=last.ckpt

    # RUN_DIR=/net/group/all-ipu/blazej_ckpt/MPNN++_BackToOG_130M_20240124_003922
    # CONFIG_NAME=MPNN_130M_LargeMix_hid512.yaml
        # RUN_NAME_SUFFIX="${FAMILY_NAME}_130M_best"
        # CKPT_NAME_FOR_TESTING=MPNN++_BackToOG_130M_20240124_003922.ckpt

        # RUN_NAME_SUFFIX="${FAMILY_NAME}_130M_last"
        # CKPT_NAME_FOR_TESTING=last.ckpt


# FAMILY_NAME="BugFixOg_last-layernorm" MUP_BASE_PATH="/net/group/all-ipu/blazej_ckpt/base_shape_10M_BackToOG/base_shape_lastnorm.yaml"
    # RUN_DIR=/net/group/all-ipu/blazej_ckpt/MPNN++_BackToOG_10M_20240125_115530
    # CONFIG_NAME=MPNN_10M_LargeMix_hid128.yaml
        # RUN_NAME_SUFFIX="${FAMILY_NAME}_10M_best"
        # CKPT_NAME_FOR_TESTING=MPNN++_BackToOG_10M_20240125_115530.ckpt

        # RUN_NAME_SUFFIX="${FAMILY_NAME}_10M_last"
        # CKPT_NAME_FOR_TESTING=last.ckpt

    # RUN_DIR=/net/group/all-ipu/blazej_ckpt/MPNN++_BackToOG_35M_20240125_115733
    # CONFIG_NAME=MPNN_35M_LargeMix_hid256.yaml
        # RUN_NAME_SUFFIX="${FAMILY_NAME}_35M_best"
        # CKPT_NAME_FOR_TESTING=MPNN++_BackToOG_35M_20240125_115733.ckpt

        # RUN_NAME_SUFFIX="${FAMILY_NAME}_35M_last"
        # CKPT_NAME_FOR_TESTING=last.ckpt

    # RUN_DIR=/net/group/all-ipu/blazej_ckpt/MPNN++_BackToOG_130M_20240125_115738
    # CONFIG_NAME=MPNN_130M_LargeMix_hid512.yaml
        # RUN_NAME_SUFFIX="${FAMILY_NAME}_130M_best"
        # CKPT_NAME_FOR_TESTING=MPNN++_BackToOG_130M_20240125_115738.ckpt

        # RUN_NAME_SUFFIX="${FAMILY_NAME}_130M_last"
        # CKPT_NAME_FOR_TESTING=last.ckpt

    # RUN_DIR=/net/group/all-ipu/blazej_ckpt/MPNN++_BackToOG_300M_20240125_115742
    # CONFIG_NAME=MPNN_300M_LargeMix_hid768.yaml
    #     RUN_NAME_SUFFIX="${FAMILY_NAME}_300M_best"
    #     CKPT_NAME_FOR_TESTING=MPNN++_BackToOG_300M_20240125_115742.ckpt

# FAMILY_NAME="BugFixOg_5e-1-g25-loss" MUP_BASE_PATH="/net/group/all-ipu/blazej_ckpt/MPNN++_BackToOG_10M_20240115_144553/base_shape.yaml"
    # RUN_DIR=/net/group/all-ipu/blazej_ckpt/MPNN++_BackToOG_10M_20240125_153229
    # CONFIG_NAME=MPNN_10M_LargeMix_hid128.yaml
        # RUN_NAME_SUFFIX="${FAMILY_NAME}_10M_best"
        # CKPT_NAME_FOR_TESTING=MPNN++_BackToOG_10M_20240125_153229.ckpt

        # RUN_NAME_SUFFIX="${FAMILY_NAME}_10M_last"
        # CKPT_NAME_FOR_TESTING=last.ckpt

    # RUN_DIR=/net/group/all-ipu/blazej_ckpt/MPNN++_BackToOG_35M_20240125_153353
    # CONFIG_NAME=MPNN_35M_LargeMix_hid256.yaml
    #     RUN_NAME_SUFFIX="${FAMILY_NAME}_35M_best"
    #     CKPT_NAME_FOR_TESTING=MPNN++_BackToOG_35M_20240125_153353.ckpt

    # RUN_DIR=/net/group/all-ipu/blazej_ckpt/MPNN++_BackToOG_130M_20240125_154522
    # CONFIG_NAME=MPNN_130M_LargeMix_hid512.yaml
    #     RUN_NAME_SUFFIX="${FAMILY_NAME}_130M_best"
    #     CKPT_NAME_FOR_TESTING=MPNN++_BackToOG_130M_20240125_154522.ckpt

    # RUN_DIR=/net/group/all-ipu/blazej_ckpt/MPNN++_BackToOG_300M_20240125_154543
    # CONFIG_NAME=MPNN_300M_LargeMix_hid768.yaml
    #     RUN_NAME_SUFFIX="${FAMILY_NAME}_300M_best"
    #     CKPT_NAME_FOR_TESTING=MPNN++_BackToOG_300M_20240125_154543.ckpt
























###### BugFixOg lr=1e-4 #######
# python graphium/cli/get_final_fingerprints.py \
#     --config-path=/net/group/all-ipu/blazej_ckpt/MPNN++_BackToOG_10M_20240122_123353 \
#     --config-name=MPNN_10M_LargeMix_hid128 \
#     datamodule.args.featurization_backend=threading \
#     datamodule.args.featurization.max_num_atoms=99999 \
#     accelerator.type=cpu \
#     architecture.mup_base_path=/net/group/all-ipu/blazej_ckpt/MPNN++_BackToOG_10M_20240115_144553/base_shape.yaml \
#     trainer.model_checkpoint.dirpath=/net/group/all-ipu/blazej_ckpt/MPNN++_BackToOG_10M_20240122_123353 \
#     +ckpt_name_for_testing=MPNN++_BackToOG_10M_20240122_123353.ckpt \
#     +run_name_suffix=BugFixOg_10M_1e-4_best

# python graphium/cli/get_final_fingerprints.py \
#     --config-path=/net/group/all-ipu/blazej_ckpt/MPNN++_BackToOG_35M_20240122_124110 \
#     --config-name=MPNN_35M_LargeMix_hid256 \
#     datamodule.args.featurization_backend=threading \
#     datamodule.args.featurization.max_num_atoms=99999 \
#     accelerator.type=cpu \
#     architecture.mup_base_path=/net/group/all-ipu/blazej_ckpt/MPNN++_BackToOG_10M_20240115_144553/base_shape.yaml \
#     trainer.model_checkpoint.dirpath=/net/group/all-ipu/blazej_ckpt/MPNN++_BackToOG_35M_20240122_124110 \
#     +ckpt_name_for_testing=MPNN++_BackToOG_35M_20240122_124110.ckpt \
#     +run_name_suffix=BugFixOg_35M_1e-4_best

# python graphium/cli/get_final_fingerprints.py \
#     --config-path=/net/group/all-ipu/blazej_ckpt/MPNN++_BackToOG_130M_20240122_123422 \
#     --config-name=MPNN_130M_LargeMix_hid512 \
#     datamodule.args.featurization_backend=threading \
#     datamodule.args.featurization.max_num_atoms=99999 \
#     accelerator.type=cpu \
#     architecture.mup_base_path=/net/group/all-ipu/blazej_ckpt/MPNN++_BackToOG_10M_20240115_144553/base_shape.yaml \
#     trainer.model_checkpoint.dirpath=/net/group/all-ipu/blazej_ckpt/MPNN++_BackToOG_130M_20240122_123422 \
#     +ckpt_name_for_testing=MPNN++_BackToOG_130M_20240122_123422.ckpt \
#     +run_name_suffix=BugFixOg_130M_1e-4_best

# python graphium/cli/get_final_fingerprints.py \
#     --config-path=/net/group/all-ipu/blazej_ckpt/MPNN++_BackToOG_300M_20240122_123421 \
#     --config-name=MPNN_300M_LargeMix_hid768.yaml \
#     datamodule.args.featurization_backend=threading \
#     datamodule.args.featurization.max_num_atoms=99999 \
#     accelerator.type=cpu \
#     architecture.mup_base_path=/net/group/all-ipu/blazej_ckpt/MPNN++_BackToOG_10M_20240115_144553/base_shape.yaml \
#     trainer.model_checkpoint.dirpath=/net/group/all-ipu/blazej_ckpt/MPNN++_BackToOG_300M_20240122_123421 \
#     +ckpt_name_for_testing=MPNN++_BackToOG_300M_20240122_123421.ckpt \
#     +run_name_suffix=BugFixOg_300M_1e-4_best






###### BugFixOg g25-loss #######

# python graphium/cli/get_final_fingerprints.py \
#     --config-path=/net/group/all-ipu/blazej_ckpt/MPNN++_BackToOG_10M_20240123_161044 \
#     --config-name=fingerprint_extraction \
#     datamodule.args.featurization_backend=threading \
#     architecture.mup_base_path=/net/group/all-ipu/blazej_ckpt/MPNN++_BackToOG_10M_20240115_144553/base_shape.yaml \
#     trainer.model_checkpoint.dirpath=/net/group/all-ipu/blazej_ckpt/MPNN++_BackToOG_10M_20240123_161044 \
#     +ckpt_name_for_testing=MPNN++_BackToOG_10M_20240123_161044.ckpt \
#     +run_name_suffix=BugFixOg_10M_g25-loss

# python graphium/cli/get_final_fingerprints.py \
#     --config-path=/net/group/all-ipu/blazej_ckpt/MPNN++_BackToOG_10M_20240123_161044 \
#     --config-name=fingerprint_extraction \
#     datamodule.args.featurization_backend=threading \
#     architecture.mup_base_path=/net/group/all-ipu/blazej_ckpt/MPNN++_BackToOG_10M_20240115_144553/base_shape.yaml \
#     trainer.model_checkpoint.dirpath=/net/group/all-ipu/blazej_ckpt/MPNN++_BackToOG_10M_20240123_161044 \
#     +ckpt_name_for_testing=last.ckpt \
#     +run_name_suffix=BugFixOg_10M_g25-loss_last

# python graphium/cli/get_final_fingerprints.py \
#     --config-path=/net/group/all-ipu/blazej_ckpt/MPNN++_BackToOG_35M_20240124_002906 \
#     --config-name=MPNN_35M_LargeMix_hid256 \
#     datamodule.args.featurization_backend=threading \
#     architecture.mup_base_path=/net/group/all-ipu/blazej_ckpt/MPNN++_BackToOG_10M_20240115_144553/base_shape.yaml \
#     trainer.model_checkpoint.dirpath=/net/group/all-ipu/blazej_ckpt/MPNN++_BackToOG_35M_20240124_002906 \
#     +ckpt_name_for_testing=MPNN++_BackToOG_35M_20240124_002906.ckpt \
#     +run_name_suffix=BugFixOg_35M_g25-loss_best

# python graphium/cli/get_final_fingerprints.py \
#     --config-path=/net/group/all-ipu/blazej_ckpt/MPNN++_BackToOG_35M_20240124_002906 \
#     --config-name=MPNN_35M_LargeMix_hid256 \
#     datamodule.args.featurization_backend=threading \
#     architecture.mup_base_path=/net/group/all-ipu/blazej_ckpt/MPNN++_BackToOG_10M_20240115_144553/base_shape.yaml \
#     trainer.model_checkpoint.dirpath=/net/group/all-ipu/blazej_ckpt/MPNN++_BackToOG_35M_20240124_002906 \
#     +ckpt_name_for_testing=last.ckpt \
#     +run_name_suffix=BugFixOg_35M_g25-loss_last

# python graphium/cli/get_final_fingerprints.py \
#     --config-path=/net/group/all-ipu/blazej_ckpt/MPNN++_BackToOG_130M_20240124_002909 \
#     --config-name=MPNN_130M_LargeMix_hid512.yaml \
#     datamodule.args.featurization_backend=threading \
#     datamodule.args.featurization.max_num_atoms=99999 \
#     accelerator.type=cpu \
#     architecture.mup_base_path=/net/group/all-ipu/blazej_ckpt/MPNN++_BackToOG_10M_20240115_144553/base_shape.yaml \
#     trainer.model_checkpoint.dirpath=/net/group/all-ipu/blazej_ckpt/MPNN++_BackToOG_130M_20240124_002909 \
#     +ckpt_name_for_testing=MPNN++_BackToOG_130M_20240124_002909.ckpt \
#     +run_name_suffix=BugFixOg_130M_g25-loss_best

# python graphium/cli/get_final_fingerprints.py \
#     --config-path=/net/group/all-ipu/blazej_ckpt/MPNN++_BackToOG_300M_20240124_004310 \
#     --config-name=MPNN_300M_LargeMix_hid768.yaml \
#     datamodule.args.featurization_backend=threading \
#     datamodule.args.featurization.max_num_atoms=99999 \
#     accelerator.type=cpu \
#     architecture.mup_base_path=/net/group/all-ipu/blazej_ckpt/MPNN++_BackToOG_10M_20240115_144553/base_shape.yaml \
#     trainer.model_checkpoint.dirpath=/net/group/all-ipu/blazej_ckpt/MPNN++_BackToOG_300M_20240124_004310 \
#     +ckpt_name_for_testing=MPNN++_BackToOG_300M_20240124_004310.ckpt \
#     +run_name_suffix=BugFixOg_300M_g25-loss_best









###### BugFixOg #######

# python graphium/cli/get_final_fingerprints.py \
#     --config-path=/net/group/all-ipu/blazej_ckpt/MPNN++_BackToOG_10M_20240119_232408 \
#     --config-name=fingerprint_extraction \
#     datamodule.args.featurization_backend=threading \
#     architecture.mup_base_path=/net/group/all-ipu/blazej_ckpt/MPNN++_BackToOG_10M_20240115_144553/base_shape.yaml \
#     trainer.model_checkpoint.dirpath=/net/group/all-ipu/blazej_ckpt/MPNN++_BackToOG_10M_20240119_232408 \
#     +ckpt_name_for_testing=MPNN++_BackToOG_10M_20240119_232408.ckpt \
#     +run_name_suffix=BugFixOg_10M

# python graphium/cli/get_final_fingerprints.py \
#     --config-path=/net/group/all-ipu/blazej_ckpt/MPNN++_BackToOG_10M_20240119_232408 \
#     --config-name=fingerprint_extraction \
#     datamodule.args.featurization_backend=threading \
#     architecture.mup_base_path=/net/group/all-ipu/blazej_ckpt/MPNN++_BackToOG_10M_20240115_144553/base_shape.yaml \
#     trainer.model_checkpoint.dirpath=/net/group/all-ipu/blazej_ckpt/MPNN++_BackToOG_10M_20240119_232408 \
#     +ckpt_name_for_testing=last.ckpt \
#     +run_name_suffix=BugFixOg_10M_last

# python graphium/cli/get_final_fingerprints.py \
#     --config-path=/net/group/all-ipu/blazej_ckpt/MPNN++_BackToOG_35M_20240122_165533 \
#     --config-name=fingerprint_extraction \
#     datamodule.args.featurization_backend=threading \
#     architecture.mup_base_path=/net/group/all-ipu/blazej_ckpt/MPNN++_BackToOG_10M_20240115_144553/base_shape.yaml \
#     trainer.model_checkpoint.dirpath=/net/group/all-ipu/blazej_ckpt/MPNN++_BackToOG_35M_20240122_165533 \
#     +ckpt_name_for_testing=MPNN++_BackToOG_35M_20240122_165533.ckpt \
#     +run_name_suffix=BugFixOg_35M

# python graphium/cli/get_final_fingerprints.py \
#     --config-path=/net/group/all-ipu/blazej_ckpt/MPNN++_BackToOG_35M_20240122_165533 \
#     --config-name=fingerprint_extraction \
#     datamodule.args.featurization_backend=threading \
#     architecture.mup_base_path=/net/group/all-ipu/blazej_ckpt/MPNN++_BackToOG_10M_20240115_144553/base_shape.yaml \
#     trainer.model_checkpoint.dirpath=/net/group/all-ipu/blazej_ckpt/MPNN++_BackToOG_35M_20240122_165533 \
#     +ckpt_name_for_testing=last.ckpt \
#     +run_name_suffix=BugFixOg_35M_last

# python graphium/cli/get_final_fingerprints.py \
#     --config-path=/net/group/all-ipu/blazej_ckpt/MPNN++_BackToOG_130M_20240122_131731 \
#     --config-name=MPNN_130M_LargeMix_hid512 \
#     datamodule.args.featurization_backend=threading \
#     architecture.mup_base_path=/net/group/all-ipu/blazej_ckpt/MPNN++_BackToOG_10M_20240115_144553/base_shape.yaml \
#     trainer.model_checkpoint.dirpath=/net/group/all-ipu/blazej_ckpt/MPNN++_BackToOG_130M_20240122_131731 \
#     +ckpt_name_for_testing=MPNN++_BackToOG_130M_20240122_131731.ckpt \
#     +run_name_suffix=BugFixOg_130M_best

# python graphium/cli/get_final_fingerprints.py \
#     --config-path=/net/group/all-ipu/blazej_ckpt/MPNN++_BackToOG_130M_20240122_131731 \
#     --config-name=MPNN_130M_LargeMix_hid512 \
#     datamodule.args.featurization_backend=threading \
#     architecture.mup_base_path=/net/group/all-ipu/blazej_ckpt/MPNN++_BackToOG_10M_20240115_144553/base_shape.yaml \
#     trainer.model_checkpoint.dirpath=/net/group/all-ipu/blazej_ckpt/MPNN++_BackToOG_130M_20240122_131731 \
#     +ckpt_name_for_testing=last.ckpt \
#     +run_name_suffix=BugFixOg_130M_last

# python graphium/cli/get_final_fingerprints.py \
#     --config-path=/net/group/all-ipu/blazej_ckpt/MPNN++_BackToOG_300M_20240120_202149 \
#     --config-name=MPNN_300M_LargeMix_hid768 \
#     datamodule.args.featurization_backend=threading \
#     datamodule.args.featurization.max_num_atoms=99999 \
#     accelerator.type=cpu \
#     architecture.mup_base_path=/net/group/all-ipu/blazej_ckpt/MPNN++_BackToOG_10M_20240115_144553/base_shape.yaml \
#     trainer.model_checkpoint.dirpath=/net/group/all-ipu/blazej_ckpt/MPNN++_BackToOG_300M_20240120_202149 \
#     +ckpt_name_for_testing=MPNN++_BackToOG_300M_20240120_202149.ckpt \
#     +run_name_suffix=BugFixOg_300M_best





















##### Before bug fix



# python graphium/cli/get_final_fingerprints.py \
#     --config-path=/net/group/all-ipu/blazej_ckpt/MPNN++_BackToOG_10M_20240115_144553 \
#     --config-name=fingerprint_extraction \
#     datamodule.args.featurization_backend=threading \
#     trainer.model_checkpoint.dirpath=/net/group/all-ipu/blazej_ckpt/MPNN++_BackToOG_10M_20240115_144553 \
#     architecture.mup_base_path=/net/group/all-ipu/blazej_ckpt/MPNN++_BackToOG_10M_20240115_144553/base_shape.yaml \
#     +ckpt_name_for_testing=MPNN++_BackToOG_10M_20240115_144553.ckpt \
#     +run_name_suffix=BackToOg_10M_incl-node-feat_best



# python graphium/cli/get_final_fingerprints.py \
#     --config-path=/net/group/all-ipu/blazej_ckpt/MPNN++_BackToOG_10M_20240115_144553 \
#     --config-name=fingerprint_extraction \
#     datamodule.args.featurization_backend=threading \
#     trainer.model_checkpoint.dirpath=/net/group/all-ipu/blazej_ckpt/MPNN++_BackToOG_10M_20240115_144553 \
#     architecture.mup_base_path=/net/group/all-ipu/blazej_ckpt/MPNN++_BackToOG_10M_20240115_144553/base_shape.yaml \
#     +ckpt_name_for_testing=last.ckpt \
#     +run_name_suffix=BackToOg_10M_easy-th_mup_last \

# python graphium/cli/get_final_fingerprints.py \
#     --config-path=/net/group/all-ipu/blazej_ckpt/MPNN++_BackToOG_35M_20240115_144555 \
#     --config-name=fingerprint_extraction \
#     datamodule.args.featurization_backend=threading \
#     architecture.mup_base_path=/net/group/all-ipu/blazej_ckpt/MPNN++_BackToOG_10M_20240115_144553/base_shape.yaml \
#     trainer.model_checkpoint.dirpath=/net/group/all-ipu/blazej_ckpt/MPNN++_BackToOG_35M_20240115_144555 \
#     +ckpt_name_for_testing=MPNN++_BackToOG_35M_20240115_144555.ckpt \
#     +run_name_suffix=BackToOg_35M_easy-th_mup

# python graphium/cli/get_final_fingerprints.py \
#     --config-path=/net/group/all-ipu/blazej_ckpt/MPNN++_BackToOG_130M_20240115_144552 \
#     --config-name=fingerprint_extraction \
#     datamodule.args.featurization_backend=threading \
#     architecture.mup_base_path=/net/group/all-ipu/blazej_ckpt/MPNN++_BackToOG_10M_20240115_144553/base_shape.yaml \
#     trainer.model_checkpoint.dirpath=/net/group/all-ipu/blazej_ckpt/MPNN++_BackToOG_130M_20240115_144552 \
#     +ckpt_name_for_testing=MPNN++_BackToOG_130M_20240115_144552.ckpt \
#     +run_name_suffix=BackToOg_130M_easy-th_mup

# python graphium/cli/get_final_fingerprints.py \
#     --config-path=/net/group/all-ipu/blazej_ckpt/MPNN++_BackToOG_300M_20240116_170808 \
#     --config-name=fingerprint_extraction \
#     datamodule.args.featurization_backend=threading \
#     architecture.mup_base_path=/net/group/all-ipu/blazej_ckpt/MPNN++_BackToOG_10M_20240115_144553/base_shape.yaml \
#     trainer.model_checkpoint.dirpath=/net/group/all-ipu/blazej_ckpt/MPNN++_BackToOG_300M_20240116_170808 \
#     +ckpt_name_for_testing=MPNN++_BackToOG_300M_20240116_170808.ckpt \
#     +run_name_suffix=BackToOg_300M_easy-th_mup





# python graphium/cli/get_final_fingerprints.py \
#     --config-path=/net/group/all-ipu/blazej_ckpt/BackToOg_10M/ \
#     --config-name=fingerprint_extraction \
#     datamodule.args.featurization_backend=threading \
#     trainer.model_checkpoint.dirpath=/net/group/all-ipu/blazej_ckpt/BackToOg_10M/ \
#     +ckpt_name_for_testing=MPNN++_BackToOG_10M_20240105_222012.ckpt \
#     +run_name_suffix=BackToOg_10M

# python graphium/cli/get_final_fingerprints.py \
#     --config-path=/net/group/all-ipu/blazej_ckpt/BackToOg_35M/ \
#     --config-name=fingerprint_extraction \
#     datamodule.args.featurization_backend=threading \
#     trainer.model_checkpoint.dirpath=/net/group/all-ipu/blazej_ckpt/BackToOg_35M/ \
#     +ckpt_name_for_testing=MPNN++_BackToOG_35M_20240105_222027.ckpt \
#     +run_name_suffix=BackToOg_35M

# python graphium/cli/get_final_fingerprints.py \
#     --config-path=/net/group/all-ipu/blazej_ckpt/BackToOg_130M/ \
#     --config-name=fingerprint_extraction \
#     datamodule.args.featurization_backend=threading \
#     trainer.model_checkpoint.dirpath=/net/group/all-ipu/blazej_ckpt/BackToOg_130M/ \
#     +ckpt_name_for_testing=MPNN++_BackToOG_130M_20240105_222017.ckpt \
#     +run_name_suffix=BackToOg_130M \
#     +architecture.mup_base_path=/net/group/all-ipu/blazej_ckpt/MPNN++_mup_150_N4crazy/mup_base_params_dominiquempnn.yaml






# python graphium/cli/get_final_fingerprints.py \
#     --config-path=/net/group/all-ipu/blazej_ckpt/MPNN++_40M_TaskHead128_easyTH/ \
#     --config-name=fingerprint_extraction \
#     datamodule.args.featurization_backend=threading \
#     trainer.model_checkpoint.dirpath=/net/group/all-ipu/blazej_ckpt/MPNN++_40M_TaskHead128_easyTH/ \
#     +ckpt_name_for_testing=MPNN++_KerstinsNewDimensions_38M_TaskHead128_easyTH_20240103_115359.ckpt \
#     +run_name_suffix=40M-MPNN-easy-th

# python graphium/cli/get_final_fingerprints.py \
#     --config-path=/net/group/all-ipu/blazej_ckpt/GPS++_40M_TaskHead128/ \
#     --config-name=fingerprint_extraction \
#     datamodule.args.featurization_backend=threading \
#     trainer.model_checkpoint.dirpath=/net/group/all-ipu/blazej_ckpt/GPS++_40M_TaskHead128 \
#     +ckpt_name_for_testing=GPS++_KerstinsNewDimensions_38M_TaskHead128_20240103_124033.ckpt \
#     +run_name_suffix=38M-GPS++