from stocknet.datasets.finance import ClientDataset, FrameConvertDataset
from stocknet.datasets.highlow import HighLowDataset
from stocknet.datasets.shift import ShiftDataset

available_dataset = {
    ClientDataset.key: ClientDataset,
    FrameConvertDataset.key: FrameConvertDataset,
    ShiftDataset.key: ShiftDataset,
    HighLowDataset.key: HighLowDataset,
}
