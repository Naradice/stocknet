import copy
import datetime
import json
import os
import sys
import unittest

module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(module_path)
import stocknet.datasets as ds

finance_client_module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../finance_client"))
sys.path.append(finance_client_module_path)
import torch

import finance_client as fc

file_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../finance_client/finance_client/data_source/mt5/OANDA-Japan MT5 Live/mt5_USDJPY_d1.csv")
)
ohlc_columns = ["high", "low", "open", "close"]


class TestDatasets(unittest.TestCase):
    def __check_minmax(self, input, output):
        # check if input and output values are less than 1 and greater than 1
        self.assertLessEqual(input.max(), 1)
        self.assertLessEqual(output.max(), 1)
        self.assertGreaterEqual(input.min(), -1)
        self.assertGreaterEqual(output.min(), -1)

    def test_ohlc(self):
        processes = [fc.utils.MinMaxPreProcess(scale=(-1, 1))]
        data_client = fc.CSVClient(file=file_path, frame=60 * 24, date_column="time", post_process=processes, columns=ohlc_columns)
        observationLength = 60
        dataset = ds.OHLCDataset(data_client=data_client, observationLength=observationLength, isTraining=True)
        i, o = dataset[0]

        self.__check_minmax(i, o)

        # check output size
        self.assertEqual(i.shape[0], observationLength)
        self.assertEqual(i.shape[1], len(ohlc_columns))
        self.assertEqual(o.shape[0], observationLength)
        self.assertEqual(o.shape[1], len(ohlc_columns))

        first_input = copy.copy(i)

        # check output size
        batch_size = 30
        i, o = dataset[0:batch_size]
        self.assertEqual(i.shape[0], batch_size)
        self.assertEqual(o.shape[0], batch_size)
        self.assertEqual(i.shape[1], observationLength)
        self.assertEqual(i.shape[2], len(ohlc_columns))
        self.assertEqual(o.shape[1], observationLength)
        self.assertEqual(o.shape[2], len(ohlc_columns))

        # check if input and output are same
        for b in range(0, batch_size):
            for index in range(0, observationLength):
                for column in range(0, len(ohlc_columns)):
                    self.assertEqual(i[b][index][column], o[b][index][column])

        data_client = fc.CSVClient(file=file_path, frame=60 * 24, date_column="time", post_process=processes, columns=ohlc_columns)
        observationLength = 60
        dataset = ds.OHLCDataset(data_client=data_client, observationLength=observationLength, isTraining=True)
        i, o = dataset[0]

        # check if output have consistency with same seed
        for index in range(0, observationLength):
            self.assertEqual(i[index][0], first_input[index][0])

        data_client = fc.CSVClient(file=file_path, frame=60 * 24, date_column="time", post_process=processes, columns=ohlc_columns)
        observationLength = 60
        dataset = ds.OHLCDataset(data_client=data_client, observationLength=observationLength, isTraining=True, seed=0)
        i, o = dataset[0]

        for index in range(0, observationLength):
            self.assertNotEqual(i[index][0], first_input[index][0])

        # print("ohlc test complete!")

    def test_ohlc_merged(self):
        processes = [fc.utils.MinMaxPreProcess(scale=(-1, 1))]
        data_client = fc.CSVClient(file=file_path, frame=60 * 24, date_column="time", post_process=processes, columns=ohlc_columns)
        observationLength = 60
        dataset = ds.OHLCDataset(data_client=data_client, observationLength=observationLength, merge_columns=True)
        i, o = dataset[0]

        self.__check_minmax(i, o)

        # check output size
        self.assertEqual(i.shape[0], observationLength * len(ohlc_columns))
        self.assertEqual(o.shape[0], observationLength * len(ohlc_columns))

        # check output size
        batch_size = 30
        i, o = dataset[0:batch_size]
        self.assertEqual(i.shape[0], batch_size)
        self.assertEqual(o.shape[0], batch_size)
        self.assertEqual(i.shape[1], observationLength * len(ohlc_columns))
        self.assertEqual(o.shape[1], observationLength * len(ohlc_columns))

    def test_shift(self):
        processes = [fc.utils.MinMaxPreProcess(scale=(-1, 1))]
        data_client = fc.CSVClient(file=file_path, frame=60 * 24, date_column="time", post_process=processes, columns=ohlc_columns)
        observationLength = 60
        out_columns = ohlc_columns.copy()
        out_columns.pop(0)
        shift = 1
        dataset = ds.ShiftDataset(
            data_client=data_client,
            observationLength=observationLength,
            in_columns=ohlc_columns,
            out_columns=out_columns,
            shift=shift,
            isTraining=True,
        )
        i, o = dataset[0]

        self.__check_minmax(i, o)

        # check output size
        self.assertEqual(i.shape[0], observationLength)
        self.assertEqual(i.shape[1], len(ohlc_columns))
        self.assertEqual(o.shape[0], len(out_columns))

        # check if output is shifted
        org_index = dataset.getActialIndex(0)
        column_index = 0
        for column in out_columns:
            self.assertEqual(dataset.data[column].iloc[org_index], o[column_index])
            column_index += 1

        shift = 2
        dataset = ds.ShiftDataset(
            data_client=data_client,
            observationLength=observationLength,
            in_columns=ohlc_columns,
            out_columns=out_columns,
            shift=shift,
            isTraining=True,
        )
        i, o = dataset[0]

        org_index = dataset.getActialIndex(0)
        column_index = 0
        for column in out_columns:
            self.assertEqual(dataset.data[column].iloc[org_index + 1], o[column_index])
            column_index += 1

        # check output size
        batch_size = 30
        i, o = dataset[0:batch_size]
        self.assertEqual(i.shape[0], batch_size)
        self.assertEqual(i.shape[1], observationLength)
        self.assertEqual(i.shape[2], len(ohlc_columns))

        self.assertEqual(o.shape[0], batch_size)
        self.assertEqual(o.shape[1], len(out_columns))

    def test_shift_merged(self):
        processes = [fc.utils.MinMaxPreProcess(scale=(-1, 1))]
        data_client = fc.CSVClient(file=file_path, frame=60 * 24, date_column="time", post_process=processes, columns=ohlc_columns)
        observationLength = 60
        out_columns = ohlc_columns.copy()
        out_columns.pop(0)
        shift = 1
        dataset = ds.ShiftDataset(
            data_client=data_client,
            observationLength=observationLength,
            in_columns=ohlc_columns,
            out_columns=out_columns,
            shift=shift,
            merge_input_columns=True,
        )
        i, o = dataset[0]

        # check if output is shifted
        # for index in range(1, len(i)):
        #    self.assertEqual(i[index], o[index])
        # check output size
        self.assertEqual(i.shape[0], observationLength * len(ohlc_columns))
        self.assertEqual(o.shape[0], len(out_columns))

        # check output size
        batch_size = 30
        i, o = dataset[0:batch_size]
        self.assertEqual(i.shape[0], batch_size)
        self.assertEqual(i.shape[1], observationLength * len(ohlc_columns))

        self.assertEqual(o.shape[0], batch_size)
        self.assertEqual(o.shape[1], len(out_columns))

    def test_highlow(self):
        mm = fc.utils.MinMaxPreProcess(scale=(-1, 1))
        processes = [mm]
        data_client = fc.CSVClient(file=file_path, frame=60 * 24, date_column="time", post_process=processes, columns=ohlc_columns)
        observationLength = 60
        out_columns = ["high", "low", "close"]
        compare_with = "close"
        dataset = ds.HighLowDataset(
            data_client=data_client,
            observationLength=observationLength,
            in_columns=ohlc_columns,
            out_columns=out_columns,
            compare_with=compare_with,
            merge_columns=True,
        )
        i, o = dataset[0]

        # check if output is shifted
        # for index in range(1, len(i)):
        #    self.assertEqual(i[index], o[index])
        # check output size
        self.assertEqual(i.shape[0], observationLength * len(ohlc_columns))
        self.assertEqual(o.shape[0], len(out_columns))

        # check output size
        batch_size = 30
        i, o = dataset[0:batch_size]
        self.assertEqual(i.shape[0], batch_size)
        self.assertEqual(i.shape[1], observationLength * len(ohlc_columns))

        self.assertEqual(o.shape[0], batch_size)
        self.assertEqual(o.shape[1], len(out_columns))

        for index in range(0, 30):
            org_index = dataset.getActialIndex(index)
            last_values = mm.revert(dataset.data[ohlc_columns].iloc[org_index - 1])
            last_value = last_values[ohlc_columns.index(compare_with)]
            next_values = mm.revert(dataset.data[ohlc_columns].iloc[org_index])

            outputs = o[index].to("cpu").detach().numpy().copy()
            column_index = 0
            for o_column in out_columns:
                next_value = next_values[ohlc_columns.index(o_column)]
                # o.shape: (30, 3)
                output = outputs[column_index]
                if output == 1:  # greater than last_value
                    self.assertGreater(next_value, last_value)
                else:
                    self.assertLessEqual(next_value, last_value)
                column_index += 1

    def test_highlow_possibility(self):
        mm = fc.utils.MinMaxPreProcess(scale=(-1, 1))
        processes = [mm]
        data_client = fc.CSVClient(file=file_path, frame=60 * 24, date_column="time", post_process=processes, columns=ohlc_columns)
        observationLength = 60
        out_columns = ["high", "low", "close"]
        compare_with = "close"
        dataset = ds.HighLowDataset(
            data_client=data_client,
            observationLength=observationLength,
            in_columns=ohlc_columns,
            out_columns=out_columns,
            compare_with=compare_with,
            merge_columns=True,
            binary_mode=False,
        )
        i, o = dataset[0]

        # check if output is shifted
        # for index in range(1, len(i)):
        #    self.assertEqual(i[index], o[index])
        # check output size
        self.assertEqual(i.shape[0], observationLength * len(ohlc_columns))
        self.assertEqual(o.shape[0], len(out_columns))

        # check output size
        batch_size = 30
        i, o = dataset[0:batch_size]
        self.assertEqual(i.shape[0], batch_size)
        self.assertEqual(i.shape[1], observationLength * len(ohlc_columns))

        self.assertEqual(o.shape[0], batch_size)
        self.assertEqual(o.shape[1], len(out_columns))

        for index in range(0, 30):
            index_out = o[index]
            totals = torch.sum(index_out, axis=1)
            for value in totals:
                self.assertEqual(value, 1)
            sf = torch.nn.Softmax(dim=1)(index_out)
            org_index = dataset.getActialIndex(index)
            last_values = mm.revert(dataset.data[ohlc_columns].iloc[org_index - 1])
            last_value = last_values[ohlc_columns.index(compare_with)]
            next_values = mm.revert(dataset.data[ohlc_columns].iloc[org_index])

            outputs = index_out.to("cpu").detach().numpy().copy()
            column_index = 0
            for o_column in out_columns:
                next_value = next_values[ohlc_columns.index(o_column)]
                # o.shape: (30, 3)
                output = outputs[column_index][0]
                if output == 1:
                    self.assertGreater(next_value, last_value)
                else:
                    self.assertLessEqual(next_value, last_value)
                column_index += 1


if __name__ == "__main__":
    unittest.main()
