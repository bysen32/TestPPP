from .aircraft_dataset import AircraftDataset
from .bird_dataset import BirdDataset

def get_trainval_dataset(args):
    if args.tag == 'aircraft':
        return AircraftDataset(phase='train', **args.__dict__), AircraftDataset(phase='val', **args.__dict__)
    elif args.tag == 'bird':
        return BirdDataset(phase='train', **args.__dict__), BirdDataset(phase="val", **args.__dict__)
    else:
        raise ValueError('Unsupported Tag {}'.format(args.tag))