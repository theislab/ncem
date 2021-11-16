import pytest

from ncem.unit_test.directories import DATA_PATH_ZHANG, DATA_PATH_JAROSCH, DATA_PATH_HARTMANN, DATA_PATH_SCHUERCH, \
    DATA_PATH_LU


class HelperTestDataLoader:
    data_origin: str

    def get_dataloader(self, data_origin: str):
        if data_origin == "zhang":
            from ncem.data import DataLoaderZhang as DataLoader

            radius = 100
            label_selection = []
            data_path = DATA_PATH_ZHANG
        elif data_origin == "jarosch":
            from ncem.data import DataLoaderJarosch as DataLoader

            radius = 100
            label_selection = []
            data_path = DATA_PATH_JAROSCH
        elif data_origin == "hartmann":
            from ncem.data import DataLoaderHartmann as DataLoader

            radius = 100
            label_selection = ["Diagnosis", "Age", "Sex"]
            data_path = DATA_PATH_HARTMANN
        elif data_origin == "pascualreguant":
            from ncem.data import DataLoaderPascualReguant as DataLoader

            radius = 100
            label_selection = []
            data_path = None  # TODO
        elif data_origin == "schuerch":
            from ncem.data import DataLoaderSchuerch as DataLoader

            radius = 100
            label_selection = ["Group"]
            data_path = DATA_PATH_SCHUERCH
        elif data_origin == "luwt":
            from ncem.data import DataLoaderSchuerch as DataLoader

            radius = 100
            label_selection = ["Group"]
            data_path = DATA_PATH_LU
        else:
            assert False

        self.data = DataLoader(data_path=data_path, radius=radius, label_selection=label_selection)
        self.data_origin = data_origin


@pytest.mark.parametrize("dataset", ["luwt", "zhang", "jarosch", "hartmann", "lu"])
def test_data_types(dataset):
    loader = HelperTestDataLoader
    loader.get_dataloader(data_origin=dataset)
