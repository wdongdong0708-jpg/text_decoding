from pathlib import Path

import numpy as np

from eeg_keyword_decoding.io import BrainVisionReader, parse_vhdr


def test_brainvision_reader_preserves_channel_time_orientation(tmp_path: Path):
    vhdr = tmp_path / "recording.vhdr"
    eeg = tmp_path / "recording.eeg"
    vhdr.write_text(
        "\n".join(
            [
                "Brain Vision Data Exchange Header File Version 1.0",
                "[Common Infos]",
                "DataFile=recording.eeg",
                "DataFormat=BINARY",
                "DataOrientation=MULTIPLEXED",
                "NumberOfChannels=2",
                "SamplingInterval=4000",
                "[Binary Infos]",
                "BinaryFormat=IEEE_FLOAT_32",
            ]
        ),
        encoding="utf-8",
    )
    multiplexed = np.array(
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
        ],
        dtype="<f4",
    )
    multiplexed.tofile(eeg)

    info = parse_vhdr(vhdr)
    reader = BrainVisionReader(vhdr)

    assert info.sfreq == 250.0
    assert reader.n_channels == 2
    assert reader.n_samples == 3
    np.testing.assert_array_equal(
        reader.read_window(1, 3),
        np.array([[2.0, 3.0], [20.0, 30.0]], dtype=np.float32),
    )

