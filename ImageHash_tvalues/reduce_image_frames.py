# ============================================================
# IMAGE HASHING ABLATION STUDY
# ============================================================
#
# This script performs an ablation study for frame deduplication
# on 1 FPS TV news video frames from CNN, FOXNEWS, and MSNBC.
#
# The script evaluates:
#
#   1. Image-hash similarity thresholds:
#        - t = 3
#        - t = 4
#        - t = 5
#
#   2. Representative-frame selection strategies:
#        - first  : select the first frame in each group
#        - middle : select the middle frame in each group
#        - last   : select the last frame in each group
#
# This produces 9 configurations in total:
#
#   t3 + first / middle / last
#   t4 + first / middle / last
#   t5 + first / middle / last
#
# For each original .tar recording, the script:
#
#   1. Extracts the 1 FPS frames.
#   2. Computes an average perceptual hash for each frame.
#   3. Sequentially groups visually similar consecutive frames.
#   4. Starts a new group when the Hamming distance between the
#      current frame and the previous frame is greater than the
#      selected threshold.
#   5. Selects one representative frame from each group using
#      the specified strategy.
#   6. Saves the selected frames directly into a reduced .tar
#      dataset.
#
# The original frame naming convention is preserved for the
# selected representative frames:
#
#   - Single-frame group: 000123.jpg
#   - Multi-frame group:  000123-000130.jpg
#
# In addition to the 9 reduced datasets, the script generates:
#
#   group_metadata.csv
#       Detailed information for every group, including the
#       group boundaries and selected representative frame.
#
#   summary.csv
#       Per-recording statistics such as input frames, retained
#       frames, reduction percentage, and group-size statistics.
#
#   ALL_CHANNELS_summary.csv
#       Combined summary results for CNN, FOXNEWS, and MSNBC.
#
# The resulting datasets are organized as:
#
#   reduced_image_data_ablation/
#       CNN/
#           t3/first/
#           t3/middle/
#           t3/last/
#           t4/first/
#           ...
#       FOXNEWS/
#           ...
#       MSNBC/
#           ...
#
# The purpose of this ablation is to empirically compare different
# similarity thresholds and representative-frame strategies and
# identify the configuration that provides the best trade-off
# between reducing redundant frames and preserving frames useful
# for downstream visual/social-media detection.
#
# ============================================================



import os
import tarfile
import tempfile
from PIL import Image
import imagehash
import csv
import time


# ============================================================
# CONFIGURATION
# ============================================================

CHANNELS = ["CNN", "FOXNEWS", "MSNBC"]

INPUT_BASE = "/home/hjaya002/LLMS/image_data"
OUTPUT_BASE = "/home/hjaya002/ODU_CS_LLMs/EvaluationStudy/ReducedImageData"

HASH_SIZE = 8
THRESHOLDS = [3, 4, 5]
STRATEGIES = ["first", "middle", "last"]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_frame_number(filename):
    """
    Extract numeric frame index from filename (without extension).
    Example:
        000001.jpg -> 1
    """
    return int(os.path.splitext(filename)[0])


def find_frame_path(input_folder, frame_name):
    """
    Find the actual image file for a frame regardless of whether
    it is .jpg, .jpeg, or .png.
    """
    for ext in [".jpg", ".jpeg", ".png"]:
        path = os.path.join(input_folder, frame_name + ext)
        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        f"Could not find image file for frame: {frame_name} "
        f"in {input_folder}"
    )


def select_representative(group, strategy):
    """
    Select a representative frame from a group.

    Strategies:
        first  -> first frame
        middle -> middle frame
        last   -> last frame
    """
    if strategy == "first":
        return group[0]

    elif strategy == "middle":
        return group[len(group) // 2]

    elif strategy == "last":
        return group[-1]

    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def group_frames(frame_names, frame_hashes, threshold):
    """
    Group frames using the same sequential grouping logic as
    the original script.

    A new group starts when:

        Hamming distance > threshold

    Otherwise, the frame remains in the current group.
    """
    groups = []

    last_hash = None
    current_group = []

    for frame_name in frame_names:

        current_hash = frame_hashes[frame_name]

        if last_hash is None or last_hash - current_hash > threshold:

            # Finish previous group
            if current_group:
                groups.append(current_group)

            # Start new group
            current_group = [frame_name]

        else:
            # Continue current group
            current_group.append(frame_name)

        last_hash = current_hash

    # Finish final group
    if current_group:
        groups.append(current_group)

    return groups


def create_output_tar(
    input_folder,
    groups,
    strategy,
    output_tar_path
):
    """
    Create one output .tar containing the representative frame
    from each group.

    The representative frame is stored using the SAME naming
    convention as the previous ablation version:

        start-end.jpg

    for multi-frame groups, and:

        frame.jpg

    for single-frame groups.
    """

    os.makedirs(os.path.dirname(output_tar_path), exist_ok=True)

    with tarfile.open(output_tar_path, "w") as output_tar:

        for group in groups:

            representative_frame = select_representative(
                group,
                strategy
            )

            src_file = find_frame_path(
                input_folder,
                representative_frame
            )

            # ------------------------------------------------
            # Keep the previous output naming convention
            # ------------------------------------------------

            extension = os.path.splitext(src_file)[1]

            if len(group) > 1:
                output_name = (
                    f"{group[0]}-{group[-1]}{extension}"
                )
            else:
                output_name = (
                    f"{group[0]}{extension}"
                )

            # Add the selected frame to the tar using the
            # previous ablation output filename.
            output_tar.add(
                src_file,
                arcname=output_name
            )


# ============================================================
# PROCESS ONE RECORDING
# ============================================================

def process_recording(
    channel,
    tar_filename,
    input_folder,
    output_base_channel,
    metadata_writer,
    summary_writer
):
    """
    Process one original recording.

    For this recording:

        1. Load all frames
        2. Compute hashes once
        3. Generate groups for t=3,4,5
        4. Generate first/middle/last tar files
        5. Write metadata
        6. Write summary statistics
    """

    print()
    print("=" * 70)
    print(f"Channel:    {channel}")
    print(f"Recording:  {tar_filename}")
    print("=" * 70)

    # --------------------------------------------------------
    # Find all image files
    # --------------------------------------------------------

    image_files = sorted(
        [
            f
            for f in os.listdir(input_folder)
            if f.lower().endswith(
                (".jpg", ".jpeg", ".png")
            )
        ],
        key=get_frame_number
    )

    if not image_files:
        print("WARNING: No image files found.")
        return

    frame_names = [
        os.path.splitext(f)[0]
        for f in image_files
    ]

    total_input_frames = len(frame_names)

    print(f"Input frames: {total_input_frames:,}")

    # --------------------------------------------------------
    # Compute hashes ONCE
    # --------------------------------------------------------

    print("Computing image hashes...")

    frame_hashes = {}

    for i, filename in enumerate(image_files):

        filepath = os.path.join(
            input_folder,
            filename
        )

        with Image.open(filepath) as image:
            current_hash = imagehash.average_hash(
                image,
                hash_size=HASH_SIZE
            )

        frame_name = os.path.splitext(filename)[0]

        frame_hashes[frame_name] = current_hash

        if (i + 1) % 5000 == 0:
            print(
                f"  Hashed {i + 1:,}/{total_input_frames:,}"
            )

    print("Hashing complete.")

    # --------------------------------------------------------
    # Process each threshold
    # --------------------------------------------------------

    for threshold in THRESHOLDS:

        print()
        print(f"Processing threshold t={threshold}")

        # ----------------------------------------------------
        # Generate groups
        # ----------------------------------------------------

        groups = group_frames(
            frame_names,
            frame_hashes,
            threshold
        )

        total_groups = len(groups)

        retained_frames = total_groups

        reduction_percentage = (
            100.0
            * (1.0 - retained_frames / total_input_frames)
        )

        group_sizes = [
            len(group)
            for group in groups
        ]

        average_group_size = (
            sum(group_sizes) / len(group_sizes)
            if group_sizes
            else 0
        )

        max_group_size = (
            max(group_sizes)
            if group_sizes
            else 0
        )

        min_group_size = (
            min(group_sizes)
            if group_sizes
            else 0
        )

        print(
            f"  Groups / retained frames: "
            f"{retained_frames:,}"
        )

        print(
            f"  Reduction: "
            f"{reduction_percentage:.2f}%"
        )

        # ----------------------------------------------------
        # Process first / middle / last
        # ----------------------------------------------------

        for strategy in STRATEGIES:

            print(
                f"  Creating t{threshold} "
                f"+ {strategy} tar..."
            )

            # ------------------------------------------------
            # Output directory
            # ------------------------------------------------

            output_folder = os.path.join(
                output_base_channel,
                f"t{threshold}",
                strategy
            )

            os.makedirs(
                output_folder,
                exist_ok=True
            )

            # ------------------------------------------------
            # Output tar filename
            #
            # Keep original recording name, but create a
            # reduced .tar dataset.
            # ------------------------------------------------

            output_tar_filename = tar_filename

            output_tar_path = os.path.join(
                output_folder,
                output_tar_filename
            )

            # ------------------------------------------------
            # Create tar
            # ------------------------------------------------

            create_output_tar(
                input_folder=input_folder,
                groups=groups,
                strategy=strategy,
                output_tar_path=output_tar_path
            )

            # ------------------------------------------------
            # Metadata for every group
            # ------------------------------------------------

            for group_id, group in enumerate(
                groups,
                start=1
            ):

                representative_frame = (
                    select_representative(
                        group,
                        strategy
                    )
                )

                representative_path = (
                    find_frame_path(
                        input_folder,
                        representative_frame
                    )
                )

                representative_extension = (
                    os.path.splitext(
                        representative_path
                    )[1]
                )

                if len(group) > 1:
                    representative_filename = (
                        f"{group[0]}-{group[-1]}"
                        f"{representative_extension}"
                    )
                else:
                    representative_filename = (
                        f"{group[0]}"
                        f"{representative_extension}"
                    )

                metadata_writer.writerow({
                    "channel": channel,
                    "recording": tar_filename,
                    "threshold": threshold,
                    "strategy": strategy,
                    "group_id": group_id,
                    "group_start": group[0],
                    "group_end": group[-1],
                    "group_size": len(group),
                    "representative_frame":
                        representative_frame,
                    "representative_filename":
                        representative_filename,
                    "output_tar":
                        os.path.relpath(
                            output_tar_path,
                            OUTPUT_BASE
                        )
                })

            # ------------------------------------------------
            # Summary row
            # ------------------------------------------------

            summary_writer.writerow({
                "channel": channel,
                "recording": tar_filename,
                "threshold": threshold,
                "strategy": strategy,
                "input_frames": total_input_frames,
                "retained_frames": retained_frames,
                "reduction_percentage":
                    round(
                        reduction_percentage,
                        4
                    ),
                "num_groups": total_groups,
                "average_group_size":
                    round(
                        average_group_size,
                        4
                    ),
                "min_group_size":
                    min_group_size,
                "max_group_size":
                    max_group_size,
                "output_tar":
                    os.path.relpath(
                        output_tar_path,
                        OUTPUT_BASE
                    )
            })

            print(
                f"    Saved: {output_tar_path}"
            )

    print()
    print(f"Finished recording: {tar_filename}")


# ============================================================
# PROCESS ONE CHANNEL
# ============================================================

def process_channel(
    channel,
    metadata_writer,
    summary_writer
):
    """
    Process all recordings for one channel.
    """

    input_base_channel = os.path.join(
        INPUT_BASE,
        channel
    )

    output_base_channel = os.path.join(
        OUTPUT_BASE,
        channel
    )

    if not os.path.exists(input_base_channel):
        print(
            f"WARNING: Input directory does not exist: "
            f"{input_base_channel}"
        )
        return

    os.makedirs(
        output_base_channel,
        exist_ok=True
    )

    tar_files = sorted(
        [
            f
            for f in os.listdir(
                input_base_channel
            )
            if f.lower().endswith(".tar")
        ]
    )

    print()
    print("#" * 80)
    print(f"PROCESSING CHANNEL: {channel}")
    print(f"Number of recordings: {len(tar_files)}")
    print("#" * 80)

    for tar_index, tar_filename in enumerate(
        tar_files,
        start=1
    ):

        print()
        print(
            f"[{tar_index}/{len(tar_files)}] "
            f"{tar_filename}"
        )

        tar_path = os.path.join(
            input_base_channel,
            tar_filename
        )

        # ----------------------------------------------------
        # Preserve original folder-name logic
        # ----------------------------------------------------

        folder_name = tar_filename.split(
            ".frames1fps"
        )[0]

        # ----------------------------------------------------
        # Extract temporary copy
        # ----------------------------------------------------

        with tempfile.TemporaryDirectory() as tmp_dir:

            print("Extracting tar...")

            with tarfile.open(
                tar_path,
                "r"
            ) as tar:

                tar.extractall(
                    path=tmp_dir
                )

            extracted_folder = os.path.join(
                tmp_dir,
                folder_name
            )

            # ------------------------------------------------
            # Safety check
            # ------------------------------------------------

            if not os.path.isdir(
                extracted_folder
            ):

                print(
                    "WARNING: Expected extracted folder "
                    f"not found:\n"
                    f"{extracted_folder}"
                )

                # Try processing directly from tmp_dir
                # if the tar does not contain the expected
                # top-level folder.
                extracted_folder = tmp_dir

            # ------------------------------------------------
            # Process recording
            # ------------------------------------------------

            process_recording(
                channel=channel,
                tar_filename=tar_filename,
                input_folder=extracted_folder,
                output_base_channel=
                    output_base_channel,
                metadata_writer=
                    metadata_writer,
                summary_writer=
                    summary_writer
            )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("IMAGE HASH ABLATION")
    print("=" * 80)
    print()
    print(f"Channels:   {CHANNELS}")
    print(f"Hash size:  {HASH_SIZE}")
    print(f"Thresholds: {THRESHOLDS}")
    print(f"Strategies: {STRATEGIES}")
    print()
    print(f"Input:  {INPUT_BASE}")
    print(f"Output: {OUTPUT_BASE}")
    print()
    print("=" * 80)

    os.makedirs(
        OUTPUT_BASE,
        exist_ok=True
    )

    # --------------------------------------------------------
    # CSV paths
    # --------------------------------------------------------

    metadata_csv = os.path.join(
        OUTPUT_BASE,
        "group_metadata.csv"
    )

    summary_csv = os.path.join(
        OUTPUT_BASE,
        "summary.csv"
    )

    all_channels_summary_csv = os.path.join(
        OUTPUT_BASE,
        "ALL_CHANNELS_summary.csv"
    )

    # --------------------------------------------------------
    # CSV field names
    # --------------------------------------------------------

    metadata_fields = [
        "channel",
        "recording",
        "threshold",
        "strategy",
        "group_id",
        "group_start",
        "group_end",
        "group_size",
        "representative_frame",
        "representative_filename",
        "output_tar"
    ]

    summary_fields = [
        "channel",
        "recording",
        "threshold",
        "strategy",
        "input_frames",
        "retained_frames",
        "reduction_percentage",
        "num_groups",
        "average_group_size",
        "min_group_size",
        "max_group_size",
        "output_tar"
    ]

    # --------------------------------------------------------
    # Open CSV files
    # --------------------------------------------------------

    start_time = time.time()

    with open(
        metadata_csv,
        "w",
        newline=""
    ) as metadata_file, open(
        summary_csv,
        "w",
        newline=""
    ) as summary_file:

        metadata_writer = csv.DictWriter(
            metadata_file,
            fieldnames=metadata_fields
        )

        summary_writer = csv.DictWriter(
            summary_file,
            fieldnames=summary_fields
        )

        metadata_writer.writeheader()
        summary_writer.writeheader()

        # ----------------------------------------------------
        # Process all three channels
        # ----------------------------------------------------

        for channel in CHANNELS:

            process_channel(
                channel=channel,
                metadata_writer=
                    metadata_writer,
                summary_writer=
                    summary_writer
            )

    # --------------------------------------------------------
    # Create ALL_CHANNELS_summary.csv
    # --------------------------------------------------------

    print()
    print("Creating combined summary...")

    with open(
        summary_csv,
        "r",
        newline=""
    ) as source_file:

        reader = csv.DictReader(
            source_file
        )

        rows = list(reader)

    with open(
        all_channels_summary_csv,
        "w",
        newline=""
    ) as output_file:

        writer = csv.DictWriter(
            output_file,
            fieldnames=summary_fields
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(row)

    # --------------------------------------------------------
    # Calculate aggregate statistics
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("ABLATION COMPLETE")
    print("=" * 80)

    elapsed = time.time() - start_time

    print(
        f"Total runtime: "
        f"{elapsed / 3600:.2f} hours"
    )

    print()
    print("Output directory:")
    print(OUTPUT_BASE)

    print()
    print("CSV files:")
    print(f"  {metadata_csv}")
    print(f"  {summary_csv}")
    print(f"  {all_channels_summary_csv}")

    print()
    print("Configurations:")
    print("  t3 + first")
    print("  t3 + middle")
    print("  t3 + last")
    print("  t4 + first")
    print("  t4 + middle")
    print("  t4 + last")
    print("  t5 + first")
    print("  t5 + middle")
    print("  t5 + last")

    print()
    print("=" * 80)


if __name__ == "__main__":
    main()