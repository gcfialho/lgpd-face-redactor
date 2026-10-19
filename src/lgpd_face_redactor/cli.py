"""Command line: `lgpd-redact INPUT -o OUTPUT --keep REFERENCE.jpg`."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from lgpd_face_redactor.pipeline import RedactionConfig, redact_file
from lgpd_face_redactor.policy import PolicyConfig
from lgpd_face_redactor.types import Style


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="lgpd-redact",
        description="Blur every face except the data subject's. Without --keep, every face is blurred.",
    )
    parser.add_argument("input", type=Path, help="Video or image to redact")
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument(
        "--keep", type=Path, action="append", default=[], metavar="PHOTO",
        help="Reference photo of a person to keep visible (one face per photo). Repeat for several photos or people.",
    )  # fmt: skip
    parser.add_argument("--audit", type=Path, help="Where to write the JSON audit record (default: OUTPUT.audit.json)")
    parser.add_argument("--threshold", type=float, default=PolicyConfig.threshold, help="Track-level cosine similarity")
    parser.add_argument("--style", type=Style, choices=list(Style), default=Style.PIXELATE)
    parser.add_argument("--keep-audio", action="store_true", help="Keep the audio track (voices are personal data too)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s")

    policy = PolicyConfig(threshold=args.threshold, frame_threshold=min(PolicyConfig.frame_threshold, args.threshold))
    config = RedactionConfig(policy=policy, style=args.style, keep_audio=args.keep_audio)
    _, record = redact_file(args.input, args.output, args.keep, config)

    audit_path = args.audit or args.output.with_name(args.output.name + ".audit.json")
    audit_path.write_text(json.dumps(record, indent=2) + "\n")
    summary = record["summary"]
    print(
        f"{summary['faces_detected']} faces, {summary['tracks']} tracks: "
        f"{summary['tracks_kept']} kept, {summary['tracks_blurred']} blurred -> {args.output} (audit: {audit_path})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
