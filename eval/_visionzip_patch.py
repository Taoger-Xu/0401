def add_visionzip_args(parser):
    parser.add_argument("--vz-dominant", type=int, default=None,
                         help="VisionZip dominant token count (includes CLS token). Omit to run the vanilla baseline.")
    parser.add_argument("--vz-contextual", type=int, default=None,
                         help="VisionZip contextual token count.")


def maybe_apply_visionzip(model, args):
    if args.vz_dominant is None:
        return model
    from visionzip import visionzip
    return visionzip(model, dominant=args.vz_dominant, contextual=args.vz_contextual)
