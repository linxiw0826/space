#!/usr/bin/env python3
"""Visual quality control for ScanNet Part A Gold manifests."""

import argparse
import json
import random
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


AXIS_NAMES = ["x", "y", "z"]
PROJECTIONS = [(0, 1), (0, 2), (1, 2)]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-scenes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--pc-means",
        type=Path,
        help="Optional scene-to-translation JSON applied to bbox centroids.",
    )
    return parser.parse_args()


def load_manifest(path):
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_frames(video_path, frame_indices):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    frames = {}
    for frame_index in frame_indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = capture.read()
        if not ok:
            capture.release()
            raise ValueError(
                f"Cannot read frame {frame_index} from {video_path}"
            )
        frames[int(frame_index)] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    capture.release()
    return frames


def node_centroid(node):
    value = node["bbox_3d"].get("centroid")
    if value is None or len(value) != 3:
        raise ValueError(f"Node has invalid centroid: {node['node_id']}")
    return np.asarray(value, dtype=float)


def candidate_position(view):
    matrix = np.asarray(view["camera_to_world"], dtype=float)
    return matrix[:3, 3]


def add_frame_panel(axis, image, view, node_positions, intrinsic):
    axis.imshow(image)
    axis.axis("off")
    visible_nodes = view["visible_nodes"]
    labels = [item["node_id"] for item in visible_nodes[:7]]
    suffix = "\n".join(labels)
    if len(visible_nodes) > 7:
        suffix += f"\n+{len(visible_nodes) - 7} more"
    title = (
        f"candidate={view['candidate_index']} "
        f"frame={view['mp4_frame_index']} "
        f"visible={len(visible_nodes)}"
    )
    axis.set_title(title, fontsize=9)
    axis.text(
        0.01,
        0.99,
        suffix or "(no exact nodes visible)",
        transform=axis.transAxes,
        va="top",
        ha="left",
        fontsize=7,
        color="white",
        bbox={"facecolor": "black", "alpha": 0.55, "pad": 2},
    )
    world_to_camera = np.linalg.inv(
        np.asarray(view["camera_to_world"], dtype=float)
    )
    height, width = image.shape[:2]
    for visible in visible_nodes:
        position = node_positions.get(visible["node_id"])
        if position is None:
            continue
        camera_point = world_to_camera @ np.r_[position, 1.0]
        if camera_point[2] <= 1e-6:
            continue
        pixel = intrinsic @ camera_point
        u = pixel[0] / pixel[2]
        v = pixel[1] / pixel[2]
        if not (0 <= u < width and 0 <= v < height):
            continue
        axis.scatter(
            [u], [v], s=35, marker="x", linewidths=1.5,
            color="#ff00ff",
        )
        axis.annotate(
            visible["node_id"],
            (u, v),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=6,
            color="#ff00ff",
        )


def add_projection(axis, camera_positions, sft_mask, node_positions, nodes, dims):
    first, second = dims
    axis.plot(
        camera_positions[:, first],
        camera_positions[:, second],
        "-o",
        color="#386cb0",
        linewidth=1,
        markersize=3,
        label="32 camera poses",
    )
    axis.scatter(
        camera_positions[sft_mask, first],
        camera_positions[sft_mask, second],
        color="#e41a1c",
        marker="*",
        s=70,
        label="8 SFT views",
        zorder=5,
    )
    if len(node_positions):
        axis.scatter(
            node_positions[:, first],
            node_positions[:, second],
            color="#4daf4a",
            marker="s",
            s=25,
            label="object nodes",
        )
        for index, node in enumerate(nodes[:20]):
            axis.annotate(
                node["category"],
                (
                    node_positions[index, first],
                    node_positions[index, second],
                ),
                fontsize=6,
                alpha=0.75,
            )
    axis.set_xlabel(AXIS_NAMES[first])
    axis.set_ylabel(AXIS_NAMES[second])
    axis.set_title(
        f"{AXIS_NAMES[first]}-{AXIS_NAMES[second]} projection"
    )
    axis.axis("equal")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7)


def add_visibility_matrix(axis, item):
    nodes = item["nodes"]
    node_index = {node["node_id"]: index for index, node in enumerate(nodes)}
    matrix = np.zeros((len(item["candidate_views"]), len(nodes)), dtype=float)
    for view_index, view in enumerate(item["candidate_views"]):
        for visible in view["visible_nodes"]:
            column = node_index.get(visible["node_id"])
            if column is not None:
                matrix[view_index, column] = visible["visible_pixels"]
    axis.imshow(
        np.log1p(matrix),
        aspect="auto",
        interpolation="nearest",
        cmap="viridis",
    )
    axis.set_xlabel(f"object nodes (n={len(nodes)})")
    axis.set_ylabel("candidate view index")
    axis.set_title("log(1 + visible pixels)")
    axis.set_yticks(list(range(0, 32, 4)))


def visualize_scene(item, output_path, offset):
    sft_views = [
        view for view in item["candidate_views"] if view["is_sft_view"]
    ]
    frames = read_frames(
        item["video_path"],
        [view["mp4_frame_index"] for view in sft_views],
    )
    camera_positions = np.stack([
        candidate_position(view) for view in item["candidate_views"]
    ])
    sft_mask = np.asarray([
        bool(view["is_sft_view"]) for view in item["candidate_views"]
    ])
    nodes = item["nodes"]
    node_positions = (
        np.stack([node_centroid(node) + offset for node in nodes])
        if nodes else np.empty((0, 3))
    )
    node_position_lookup = {
        node["node_id"]: node_positions[index]
        for index, node in enumerate(nodes)
    }
    intrinsic = np.asarray(item["intrinsic_color"], dtype=float)

    figure = plt.figure(figsize=(20, 13), constrained_layout=True)
    grid = figure.add_gridspec(3, 4)

    for index, view in enumerate(sft_views):
        axis = figure.add_subplot(grid[index // 4, index % 4])
        add_frame_panel(
            axis,
            frames[view["mp4_frame_index"]],
            view,
            node_position_lookup,
            intrinsic,
        )

    for projection_index, dims in enumerate(PROJECTIONS):
        axis = figure.add_subplot(grid[2, projection_index])
        add_projection(
            axis,
            camera_positions,
            sft_mask,
            node_positions,
            nodes,
            dims,
        )

    axis = figure.add_subplot(grid[2, 3])
    add_visibility_matrix(axis, item)

    figure.suptitle(
        f"{item['scene_id']} | QA={len(item['qa'])} "
        f"| nodes={len(nodes)} | frames={item['video_frame_count']} "
        f"| offset={np.round(offset, 3).tolist()}",
        fontsize=15,
    )
    figure.savefig(output_path, dpi=140)
    plt.close(figure)


def main():
    args = parse_args()
    items = load_manifest(args.manifest)
    pc_means = {}
    if args.pc_means:
        with args.pc_means.open("r", encoding="utf-8") as handle:
            pc_means = json.load(handle)
    rng = random.Random(args.seed)
    selected = list(items)
    rng.shuffle(selected)
    selected = selected[: args.num_scenes]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "manifest": str(args.manifest),
        "seed": args.seed,
        "requested_scenes": args.num_scenes,
        "pc_means": str(args.pc_means) if args.pc_means else None,
        "selected_scene_ids": [],
        "outputs": [],
        "errors": [],
    }
    for item in selected:
        scene = item["scene_id"]
        output_path = args.output_dir / f"{scene}_qc.png"
        try:
            offset = np.asarray(
                pc_means.get(scene, [0.0, 0.0, 0.0]),
                dtype=float,
            ).reshape(3)
            visualize_scene(item, output_path, offset)
            report["selected_scene_ids"].append(scene)
            report["outputs"].append(str(output_path))
            print(f"OK {scene}: {output_path}")
        except Exception as error:
            report["errors"].append({
                "scene_id": scene,
                "error": f"{type(error).__name__}: {error}",
            })
            print(f"ERROR {scene}: {type(error).__name__}: {error}")

    report_path = args.output_dir / "qc_report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
