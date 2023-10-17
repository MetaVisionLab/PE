#!/bin/bash
if [ $3 = "PACS" ]
then
  sketch_cfg="--trainer DRT_DG_Mixup --source-domains photo cartoon art_painting --target-domains sketch --output-dir drt_dg_r/pacs_resnet50_draac_v3_strong_transform_1/pacs/sketch/1 --dataset-config-file configs/datasets/drt_dg_r/pacs_resnet50_draac_v3_strong_transform_1.yaml --config-file configs/trainers/drt_dg_r/pacs.yaml --seed 2022 "
  python3 main.py $sketch_cfg --dymodel $1 --pe_type $2
  photo_cfg="--trainer DRT_DG_Mixup --source-domains sketch cartoon art_painting --target-domains photo --output-dir drt_dg_r/pacs_resnet50_draac_v3_strong_transform_1/pacs/photo/1 --dataset-config-file configs/datasets/drt_dg_r/pacs_resnet50_draac_v3_strong_transform_1.yaml --config-file configs/trainers/drt_dg_r/pacs.yaml --seed 2022 "
  python3 main.py $photo_cfg --dymodel $1 --pe_type $2
  cartoon_cfg="--trainer DRT_DG_Mixup --source-domains sketch photo art_painting --target-domains cartoon --output-dir drt_dg_r/pacs_resnet50_draac_v3_strong_transform_1/pacs/cartoon/1 --dataset-config-file configs/datasets/drt_dg_r/pacs_resnet50_draac_v3_strong_transform_1.yaml --config-file configs/trainers/drt_dg_r/pacs.yaml --seed 2022 "
  python3 main.py $cartoon_cfg --dymodel $1 --pe_type $2
  art_painting_cfg="--trainer DRT_DG_Mixup --source-domains sketch photo cartoon --target-domains art_painting --output-dir drt_dg_r/pacs_resnet50_draac_v3_strong_transform_1/pacs/art_painting/1 --dataset-config-file configs/datasets/drt_dg_r/pacs_resnet50_draac_v3_strong_transform_1.yaml --config-file configs/trainers/drt_dg_r/pacs.yaml --seed 2022 "
  python3 main.py $art_painting_cfg --dymodel $1 --pe_type $2
  python3 parse_test_res.py "exprs/$2_$1/drt_dg_r/pacs_resnet50_draac_v3_strong_transform_1/pacs" --last-five --multi-exp
elif [ $3 = "OfficeHome" ]
then
  art_cfg="--trainer DRT_DG_Mixup --source-domains clipart product real_world --target-domains art --output-dir drt_dg_r/office_home_dg_resnet50_draac_v3_strong_transform_2/office_home/art/1 --dataset-config-file configs/datasets/drt_dg_r/office_home_dg_resnet50_draac_v3_strong_transform_2.yaml --config-file configs/trainers/drt_dg_r/office_home_dg.yaml --seed 2022 "
  python3 main.py $art_cfg --dymodel $1 --pe_type $2
  clipart_cfg="--trainer DRT_DG_Mixup --source-domains art product real_world --target-domains clipart --output-dir drt_dg_r/office_home_dg_resnet50_draac_v3_strong_transform_2/office_home/clipart/1 --dataset-config-file configs/datasets/drt_dg_r/office_home_dg_resnet50_draac_v3_strong_transform_2.yaml --config-file configs/trainers/drt_dg_r/office_home_dg.yaml --seed 2022 "
  python3 main.py $clipart_cfg --dymodel $1 --pe_type $2
  product_cfg="--trainer DRT_DG_Mixup --source-domains art clipart real_world --target-domains product --output-dir drt_dg_r/office_home_dg_resnet50_draac_v3_strong_transform_2/office_home/product/1 --dataset-config-file configs/datasets/drt_dg_r/office_home_dg_resnet50_draac_v3_strong_transform_2.yaml --config-file configs/trainers/drt_dg_r/office_home_dg.yaml --seed 2022 "
  python3 main.py $product_cfg --dymodel $1 --pe_type $2
  real_world_cfg="--trainer DRT_DG_Mixup --source-domains art clipart product --target-domains real_world --output-dir drt_dg_r/office_home_dg_resnet50_draac_v3_strong_transform_2/office_home/real_world/1 --dataset-config-file configs/datasets/drt_dg_r/office_home_dg_resnet50_draac_v3_strong_transform_2.yaml --config-file configs/trainers/drt_dg_r/office_home_dg.yaml --seed 2022 "
  python3 main.py $real_world_cfg --dymodel $1 --pe_type $2
  python3 parse_test_res.py "exprs/$2_$1/drt_dg_r/office_home_dg_resnet50_draac_v3_strong_transform_2/office_home" --last-five --multi-exp
elif [ $3 = "VLCS" ]
then
  caltech_cfg="--trainer DRT_DG_Mixup --source-domains labelme pascal sun --target-domains caltech --output-dir drt_dg_r/vlcs_resnet50_draac_v3_strong_transform2/vlcs/caltech/1 --dataset-config-file configs/datasets/drt_dg_r/vlcs_resnet50_draac_v3_strong_transform2.yaml --config-file configs/trainers/drt_dg_r/vlcs.yaml --seed 2022 "
  python3 main.py $caltech_cfg --dymodel $1 --pe_type $2
  labelme_cfg="--trainer DRT_DG_Mixup --source-domains caltech pascal sun --target-domains labelme --output-dir drt_dg_r/vlcs_resnet50_draac_v3_strong_transform2/vlcs/labelme/1 --dataset-config-file configs/datasets/drt_dg_r/vlcs_resnet50_draac_v3_strong_transform2.yaml --config-file configs/trainers/drt_dg_r/vlcs.yaml --seed 2022 "
  python3 main.py $labelme_cfg --dymodel $1 --pe_type $2
  pascal_cfg="--trainer DRT_DG_Mixup --source-domains caltech labelme sun --target-domains pascal --output-dir drt_dg_r/vlcs_resnet50_draac_v3_strong_transform2/vlcs/pascal/1 --dataset-config-file configs/datasets/drt_dg_r/vlcs_resnet50_draac_v3_strong_transform2.yaml --config-file configs/trainers/drt_dg_r/vlcs.yaml --seed 2022 "
  python3 main.py $pascal_cfg --dymodel $1 --pe_type $2
  sun_cfg="--trainer DRT_DG_Mixup --source-domains caltech labelme pascal --target-domains sun --output-dir drt_dg_r/vlcs_resnet50_draac_v3_strong_transform2/vlcs/sun/1 --dataset-config-file configs/datasets/drt_dg_r/vlcs_resnet50_draac_v3_strong_transform2.yaml --config-file configs/trainers/drt_dg_r/vlcs.yaml --seed 2022 "
  python3 main.py $sun_cfg --dymodel $1 --pe_type $2
  python3 parse_test_res.py "exprs/$2_$1/drt_dg_r/vlcs_resnet50_draac_v3_strong_transform2/vlcs" --last-five --multi-exp
elif [ $3 = "TerriaIncognita" ]
then
  location_38_cfg="--trainer DRT_DG_Mixup --source-domains location_43 location_46 location_100 --target-domains location_38 --output-dir drt_dg_r/DRT_DG_Mixup_dg_resnet50_draac_v3_strong_transform5/incognita_dg/location_38/1 --dataset-config-file configs/datasets/drt_dg_r/terra_incognita_dg_resnet50_draac_v3_strong_transform5.yaml --config-file configs/trainers/drt_dg_r/terra_incognita_dg.yaml --seed 2022 "
  python3 main.py $location_38_cfg --dymodel $1 --pe_type $2
  location_43_cfg="--trainer DRT_DG_Mixup --source-domains location_38 location_46 location_100 --target-domains location_43 --output-dir drt_dg_r/DRT_DG_Mixup_dg_resnet50_draac_v3_strong_transform5/incognita_dg/location_43/1 --dataset-config-file configs/datasets/drt_dg_r/terra_incognita_dg_resnet50_draac_v3_strong_transform5.yaml --config-file configs/trainers/drt_dg_r/terra_incognita_dg.yaml --seed 2022 "
  python3 main.py $location_43_cfg --dymodel $1 --pe_type $2
  location_46_cfg="--trainer DRT_DG_Mixup --source-domains location_38 location_43 location_100 --target-domains location_46 --output-dir drt_dg_r/DRT_DG_Mixup_dg_resnet50_draac_v3_strong_transform5/incognita_dg/location_46/1 --dataset-config-file configs/datasets/drt_dg_r/terra_incognita_dg_resnet50_draac_v3_strong_transform5.yaml --config-file configs/trainers/drt_dg_r/terra_incognita_dg.yaml --seed 2022 "
  python3 main.py $location_46_cfg --dymodel $1 --pe_type $2
  location_100_cfg="--trainer DRT_DG_Mixup --source-domains location_38 location_43 location_46 --target-domains location_100 --output-dir drt_dg_r/DRT_DG_Mixup_dg_resnet50_draac_v3_strong_transform5/incognita_dg/location_100/1 --dataset-config-file configs/datasets/drt_dg_r/terra_incognita_dg_resnet50_draac_v3_strong_transform5.yaml --config-file configs/trainers/drt_dg_r/terra_incognita_dg.yaml --seed 2022 "
  python3 main.py $location_100_cfg --dymodel $1 --pe_type $2
  python3 parse_test_res.py "exprs/$2_$1/ddrt_dg_r/DRT_DG_Mixup_dg_resnet50_draac_v3_strong_transform5/incognita_dg" --last-five --multi-exp
elif [ $3 = "DomainNet" ]
then
  clipart_cfg="--trainer DRT_DG_Mixup --source-domains infograph painting quickdraw real sketch --target-domains clipart --output-dir drt_dg_r/domainnet_resnet50_draac_v3/domainnet/clipart/1 --dataset-config-file configs/datasets/drt_dg_r/domainnet_resnet50_draac_v3.yaml --config-file configs/trainers/drt_dg_r/domainnet.yaml --seed 2022 "
  python3 main.py $clipart_cfg --dymodel $1 --pe_type $2
  infograph_cfg="--trainer DRT_DG_Mixup --source-domains clipart painting quickdraw real sketch --target-domains infograph --output-dir drt_dg_r/domainnet_resnet50_draac_v3/domainnet/infograph/1 --dataset-config-file configs/datasets/drt_dg_r/domainnet_resnet50_draac_v3.yaml --config-file configs/trainers/drt_dg_r/domainnet.yaml --seed 2022 "
  python3 main.py $infograph_cfg --dymodel $1 --pe_type $2
  painting_cfg="--trainer DRT_DG_Mixup --source-domains clipart infograph quickdraw real sketch --target-domains painting --output-dir drt_dg_r/domainnet_resnet50_draac_v3/domainnet/painting/1 --dataset-config-file configs/datasets/drt_dg_r/domainnet_resnet50_draac_v3.yaml --config-file configs/trainers/drt_dg_r/domainnet.yaml --seed 2022 "
  python3 main.py $painting_cfg --dymodel $1 --pe_type $2
  quickdraw_cfg="--trainer DRT_DG_Mixup --source-domains clipart infograph painting real sketch --target-domains quickdraw --output-dir drt_dg_r/domainnet_resnet50_draac_v3/domainnet/quickdraw/1 --dataset-config-file configs/datasets/drt_dg_r/domainnet_resnet50_draac_v3.yaml --config-file configs/trainers/drt_dg_r/domainnet.yaml --seed 2022 "
  python3 main.py $quickdraw_cfg --dymodel $1 --pe_type $2
  real_cfg="--trainer DRT_DG_Mixup --source-domains clipart infograph painting quickdraw sketch --target-domains real --output-dir drt_dg_r/domainnet_resnet50_draac_v3/domainnet/real/1 --dataset-config-file configs/datasets/drt_dg_r/domainnet_resnet50_draac_v3.yaml --config-file configs/trainers/drt_dg_r/domainnet.yaml --seed 2022 "
  python3 main.py $real_cfg --dymodel $1 --pe_type $2
  sketch_cfg="--trainer DRT_DG_Mixup --source-domains clipart infograph painting quickdraw real --target-domains sketch --output-dir drt_dg_r/domainnet_resnet50_draac_v3/domainnet/sketch/1 --dataset-config-file configs/datasets/drt_dg_r/domainnet_resnet50_draac_v3.yaml --config-file configs/trainers/drt_dg_r/domainnet.yaml --seed 2022 "
  python3 main.py $sketch_cfg --dymodel $1 --pe_type $2
  python3 parse_test_res.py "exprs/$2_$1/drt_dg_r/domainnet_resnet50_draac_v3/domainnet" --last-five --multi-exp
else
  echo 'The dataset arg should be one of "PACS", "VLCS", "OfficeHome", "TerriaIncognita", or "DomainNet"'
  echo "Usage: bash train.sh {arg1=dymodel} {arg2=pe_type} {arg3=dataset}"
  exit 1
fi