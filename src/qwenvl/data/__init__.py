import os
import re

# Define placeholders for dataset paths
CAMBRIAN_737K = {
    "annotation_path": "PATH_TO_CAMBRIAN_737K_ANNOTATION",
    "data_path": "",
}

CAMBRIAN_737K_PACK = {
    "annotation_path": f"PATH_TO_CAMBRIAN_737K_ANNOTATION_PACKED",
    "data_path": f"",
}

MP_DOC = {
    "annotation_path": "PATH_TO_MP_DOC_ANNOTATION",
    "data_path": "PATH_TO_MP_DOC_DATA",
}

CLEVR_MC = {
    "annotation_path": "PATH_TO_CLEVR_MC_ANNOTATION",
    "data_path": "PATH_TO_CLEVR_MC_DATA",
}

VIDEOCHATGPT = {
    "annotation_path": "PATH_TO_VIDEOCHATGPT_ANNOTATION",
    "data_path": "PATH_TO_VIDEOCHATGPT_DATA",
}
SPAR = {
    "annotation_path": "data/train/spar_7m.jsonl",
    "data_path": "data/media",
    "tag": "3d"
}

SPAR_234K = {
    "annotation_path": os.environ.get("SPAR_234K_ANN", "data/train/spar_234k.json"),
    "data_path": os.environ.get("GUIDE_DATA_ROOT", "data/media"),
    "tag": "3d"
}

LLAVA_HOUND = {
    "annotation_path": os.environ.get("LLAVA_HOUND_ANN", "data/train/llava_hound_255k.json"),
    "data_path": os.environ.get("GUIDE_DATA_ROOT", "data/media"),
    "tag": "2d"
}

LLAVA_HOUND_64K = {
    "annotation_path": os.environ.get("LLAVA_HOUND_64K_ANN", "data/train/llava_hound_64k.json"),
    "data_path": os.environ.get("GUIDE_DATA_ROOT", "data/media"),
    "tag": "2d"
}
SQA3D_32FRAMES = {
    "annotation_path": "data/train/sqa3d_train_32frame.json",
    "data_path": "data/media",
    "tag": "3d"
}
SQA3D_16FRAMES = {
    "annotation_path": "data/train/sqa3d_train_16frame.json",
    "data_path": "data/media",
    "tag": "3d"
}
SCANNET_DET = {
    "annotation_path": "data/train/scannet_det_train_4frames.json",
    "data_path": "data/media",
    "tag": "3d"
}

SCANREFER = {
    "annotation_path": "data/train/scanrefer_train_32frames.json",
    "data_path": "data/media",
    "tag": "3d"
}

SCAN2CAP = {
    "annotation_path": "data/train/scan2cap_train_32frames.json",
    "data_path": "data/media",
    "tag": "3d"
}
VLM3R_VSI_208K = {
    "annotation_path": "data/train/vlm_3r_scannet,scannetpp,arkitscenes_32f_208k.json",
    "data_path": "data/media",
    "tag": "2d"
}

VSI590K_SPAR = {
    "annotation_path": os.environ.get("VSI590K_SPAR_ANN", ""),
    "data_path": os.environ.get("VSI590K_DATA_ROOT", ""),
    "tag": "3d",
}

VSI590K_VIDEO = {
    "annotation_path": os.environ.get("VSI590K_VIDEO_ANN", ""),
    "data_path": os.environ.get("VSI590K_DATA_ROOT", ""),
    "tag": "3d",
}

data_dict = {
    "cambrian_737k": CAMBRIAN_737K,
    "cambrian_737k_pack": CAMBRIAN_737K_PACK,
    "mp_doc": MP_DOC,
    "clevr_mc": CLEVR_MC,
    "videochatgpt": VIDEOCHATGPT,
    "sqa3d_32frames": SQA3D_32FRAMES,
    "sqa3d_16frames": SQA3D_16FRAMES,
    "spar": SPAR,
    "spar_234k": SPAR_234K,
    "llava_hound": LLAVA_HOUND,
    "llava_hound_64k": LLAVA_HOUND_64K,
    "scannet_det": SCANNET_DET,
    "scanrefer": SCANREFER,
    "scan2cap": SCAN2CAP,
    "vlm3r_vsi_208k": VLM3R_VSI_208K,
    "vsi590k_spar": VSI590K_SPAR,
    "vsi590k_video": VSI590K_VIDEO,
}


def parse_sampling_rate(dataset_name):
    match = re.search(r"%(\d+)$", dataset_name)
    if match:
        return int(match.group(1)) / 100.0
    return 1.0


def data_list(dataset_names):
    config_list = []
    for dataset_name in dataset_names:
        sampling_rate = parse_sampling_rate(dataset_name)
        dataset_name = re.sub(r"%(\d+)$", "", dataset_name)
        if dataset_name in data_dict.keys():
            config = data_dict[dataset_name].copy()
            config["sampling_rate"] = sampling_rate
            config_list.append(config)
        else:
            raise ValueError(f"do not find {dataset_name}")
    return config_list


if __name__ == "__main__":
    dataset_names = ["cambrian_737k"]
    configs = data_list(dataset_names)
    for config in configs:
        print(config)
